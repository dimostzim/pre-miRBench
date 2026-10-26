#!/usr/bin/env python3
"""Retrain the pre-miRBench v3 model: the mean of three convolutional networks.

The script replays the three Agentomics training runs behind the iteration-19
deployment with their settings, in one process:

1. aligned network 1 (AlignedPairCNN, iteration 19, seed 42);
2. aligned network 2 (AlignedPairCNN, iteration 18, seed 73);
3. folded-grid network (FoldedGridPreMiRNACNN, iteration 11, seed 17).

All three use the same recipe: AdamW, latent mixup, batches of 256, at most 100
epochs, the checkpoint with the best validation AP, ReduceLROnPlateau on that AP
and early stopping after 15 epochs without an AP gain of 0.0001. The feature
preprocessing (MFE median and standardization) is fitted once on the training
split; the three Agentomics runs fitted the same values. A training split with
fewer than 1,000 rows runs 2 epochs, the smoke check of the Agentomics scripts.

--seeds gives the seeds of the three networks in the order above; the default
42 73 17 reproduces the original seeds.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from inference import (
    COMPONENTS,
    MODEL_CODE_FILES,
    RUNTIME_ARTIFACTS,
    configure_torch,
    load_model_code,
    resolve_device,
    run_inference,
    sha256,
)


MODEL_DIR = Path(__file__).resolve().parent
MODEL_CODE = MODEL_DIR / "model_code"
DEFAULT_SEEDS = (42, 73, 17)
SOURCE_ITERATIONS = (19, 18, 11)
CLASSIFICATION_THRESHOLD = 0.5
BATCH_SIZE = 256
MAX_EPOCHS = 100
SMOKE_ROWS = 1000
SMOKE_EPOCHS = 2
PATIENCE = 15
MIN_DELTA = 0.0001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain the pre-miRBench v3 model")
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--validation-data", required=True, type=Path)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    # The Agentomics runs trained on CUDA; cpu is for smoke tests.
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seeds", type=int, nargs=3, default=list(DEFAULT_SEEDS),
                        metavar=("ALIGNED_1", "ALIGNED_2", "FOLDED_GRID"))
    return parser.parse_args()


def load_labeled_split(features_code: Any, split_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """The Agentomics load_labeled_split, also accepting the exported `label` column."""
    frame = features_code.load_samples(split_dir / "input")
    labels = pd.read_csv(split_dir / "labels.csv", dtype={"id": str})
    label_column = "numeric_label" if "numeric_label" in labels.columns else "label"
    if "id" not in labels.columns or label_column not in labels.columns:
        raise ValueError(f"{split_dir / 'labels.csv'} needs id and label or numeric_label")
    if labels.id.isna().any() or labels.id.duplicated().any():
        raise ValueError("Labels must have unique nonmissing IDs")
    if len(frame) != len(labels) or set(frame.id) != set(labels.id):
        raise ValueError("Input and label IDs/counts must match exactly")
    y = labels.set_index("id").loc[frame.id, label_column]
    if not y.isin([0, 1]).all():
        raise ValueError("Labels must be binary 0/1")
    return frame, y.to_numpy(dtype=np.float32)


def seed_everything(seed: int) -> np.random.Generator:
    # Seeding of the Agentomics scripts; the returned generator draws the mixup weights.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def batch_inputs(data, structures, idx, device, aligned, build_aligned_features):
    x, mfe, partners = [a[idx].to(device) for a in data]
    if not aligned:
        return x, mfe, partners
    aligned_x, aligned_partners = build_aligned_features(x, partners, [structures[j] for j in idx.tolist()])
    return aligned_x, mfe, aligned_partners


def predict(network, data, structures, device, aligned, build_aligned_features) -> torch.Tensor:
    network.eval()
    results = []
    with torch.inference_mode():
        for idx in torch.arange(len(data[0])).split(BATCH_SIZE):
            batch = batch_inputs(data, structures, idx, device, aligned, build_aligned_features)
            results.append(network(*batch).cpu())
    return torch.cat(results)


def train_network(
    name: str,
    network_class: type,
    aligned: bool,
    seed: int,
    train: tuple,
    validation: tuple,
    device: torch.device,
    build_aligned_features: Any,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    """The training loop of the Agentomics iterations 11, 18 and 19, which differ only in the network."""
    data, structures, y = train
    validation_data, validation_structures, validation_y = validation
    rng = seed_everything(seed)
    network = network_class().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.0005, weight_decay=0.001, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, threshold=1e-4, threshold_mode="rel",
        min_lr=1e-5, cooldown=0, eps=1e-8,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    targets_all = torch.from_numpy(y).float()
    validation_targets = torch.from_numpy(validation_y).float()
    max_epochs = SMOKE_EPOCHS if len(y) < SMOKE_ROWS else MAX_EPOCHS
    best, significant, stale = -1.0, -1.0, 0
    best_state, best_epoch, history = None, None, []
    for epoch in range(1, max_epochs + 1):
        start = time.monotonic()
        network.train()
        total = 0.0
        for idx in torch.randperm(len(y)).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            lam = float(rng.beta(0.2, 0.2))
            lam = max(lam, 1 - lam)
            batch = batch_inputs(data, structures, idx, device, aligned, build_aligned_features)
            targets = targets_all[idx].to(device)
            logits, targets = network.forward_train(*batch, targets, lam, torch.randperm(len(idx), device=device))
            loss = loss_fn(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(idx)
        logits = predict(network, validation_data, validation_structures, device, aligned, build_aligned_features)
        ap = float(average_precision_score(validation_y, logits.sigmoid().numpy()))
        validation_loss = loss_fn(logits, validation_targets).item()
        if ap > best:
            best, best_epoch = ap, epoch
            best_state = {key: value.detach().clone() for key, value in network.state_dict().items()}
        if ap >= significant + MIN_DELTA:
            significant, stale = ap, 0
        else:
            stale += 1
        scheduler.step(ap)
        history.append(dict(epoch=epoch, AP=ap, loss=validation_loss, training_loss=total / len(y),
                            seconds=time.monotonic() - start, lr=optimizer.param_groups[0]["lr"]))
        print(f"{name} epoch {epoch}: validation AP {ap:.6f}, best {best:.6f} (epoch {best_epoch}), "
              f"patience left {PATIENCE - stale}", flush=True)
        if stale >= PATIENCE:
            break
    network.load_state_dict(best_state, strict=True)
    selected = predict(network, validation_data, validation_structures, device, aligned, build_aligned_features)
    summary = dict(seed=seed, class_name=network_class.__name__, epochs_run=len(history),
                   selected_epoch=best_epoch, validation_ap=best, history=history)
    del network, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_state, selected, summary


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty artifacts directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_manifest(artifacts: Path, seeds: list[int], selected_epochs: list[int]) -> None:
    manifest = {
        "format_version": 1,
        "components": [
            {"file": name, "class": cls, "view": view, "seed": seed, "selected_epoch": epoch}
            for (name, cls, view), seed, epoch in zip(COMPONENTS, seeds, selected_epochs)
        ],
        "aggregation": "float32 mean of the three sigmoid probabilities, then float64",
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "artifact_sha256": {name: sha256(artifacts / name) for name in RUNTIME_ARTIFACTS},
    }
    (artifacts / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    prepare_output(args.artifacts_dir)
    device = resolve_device(args.device)
    configure_torch()
    modules = load_model_code(MODEL_CODE)
    features_code = modules["features"]
    build_aligned_features = modules["aligned_features"].build_aligned_features
    classes = {"AlignedPairCNN": modules["aligned_model"].AlignedPairCNN,
               "FoldedGridPreMiRNACNN": modules["model"].FoldedGridPreMiRNACNN}

    frame, y = load_labeled_split(features_code, args.train_data)
    validation_frame, validation_y = load_labeled_split(features_code, args.validation_data)
    if set(frame.id) & set(validation_frame.id):
        raise ValueError("Training and validation splits share IDs")
    for name, labels in (("training", y), ("validation", validation_y)):
        if set(labels) != {0.0, 1.0}:
            raise ValueError(f"The {name} split must contain both classes")
    features = features_code.RNAFeatures().fit(frame)
    train = (tuple(torch.from_numpy(a) for a in features.transform(frame)), frame.structure.tolist(), y)
    validation = (tuple(torch.from_numpy(a) for a in features.transform(validation_frame)),
                  validation_frame.structure.tolist(), validation_y)

    states, validation_logits, summaries = [], [], []
    for (name, cls, view), seed in zip(COMPONENTS, args.seeds):
        state, logits, summary = train_network(
            name.removesuffix(".pt"), classes[cls], view == "aligned", seed, train, validation, device,
            build_aligned_features,
        )
        states.append(state)
        validation_logits.append(logits)
        summaries.append(summary)

    artifacts = args.artifacts_dir
    for name in MODEL_CODE_FILES:
        shutil.copy2(MODEL_CODE / name, artifacts / name)
    features.save(artifacts / "preprocessing.json")
    for (name, _, _), state, summary in zip(COMPONENTS, states, summaries):
        torch.save({"state_dict": state, "epoch": summary["selected_epoch"]}, artifacts / name)
    write_manifest(artifacts, args.seeds, [summary["selected_epoch"] for summary in summaries])

    # The packaged inference must give exactly the mixture of the selected checkpoints, as the
    # Agentomics scripts check.
    mixture = torch.stack(validation_logits).to(device).sigmoid().mean(0).cpu().numpy().astype(np.float64)
    with tempfile.TemporaryDirectory(dir=artifacts.parent) as scratch:
        packaged = run_inference(args.validation_data / "input", Path(scratch) / "validation.csv", artifacts, device.type)
    if not np.array_equal(packaged.id.to_numpy(), validation_frame.id.to_numpy()) or not np.array_equal(
        packaged.probability_1.to_numpy(), mixture
    ):
        raise RuntimeError("Packaged validation predictions differ from the trained networks")

    training_history = {
        "device": str(device),
        "seeds": dict(zip((name.removesuffix(".pt") for name, _, _ in COMPONENTS), args.seeds)),
        "source_iterations": dict(zip((name.removesuffix(".pt") for name, _, _ in COMPONENTS), SOURCE_ITERATIONS)),
        "training_rows": len(y),
        "validation_rows": len(validation_y),
        "preprocessing": features.stats,
        "networks": {name.removesuffix(".pt"): summary for (name, _, _), summary in zip(COMPONENTS, summaries)},
        "validation_ap": {
            **{name.removesuffix(".pt"): float(average_precision_score(validation_y, logits.sigmoid().numpy()))
               for (name, _, _), logits in zip(COMPONENTS, validation_logits)},
            "mean": float(average_precision_score(validation_y, mixture)),
        },
    }
    (artifacts / "training_history.json").write_text(json.dumps(training_history, indent=2, sort_keys=True) + "\n")
    print(json.dumps(training_history["validation_ap"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
