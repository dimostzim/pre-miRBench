#!/usr/bin/env python3
"""Retrain the fixed three-component pre-miRBench model."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from inference import DEFAULT_ARTIFACTS, load_module, resolve_device


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = MODEL_DIR / "retrained_artifacts"
PAIR_SEED = 8_675_309
SPECIES_RC_SEED = 57_721
SPECIES_SEED = 271_828
EMA_DECAY = 0.999
PATIENCE = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain the pre-miRBench model")
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--validation-data", required=True, type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--pair-batch-size", type=int, default=512)
    parser.add_argument("--species-batch-size", type=int, default=256)
    return parser.parse_args()


def seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = device.type == "cuda"


def deterministic_reverse_complement(seed: int, epoch: int, index: int) -> bool:
    value = (seed + 0x9E3779B97F4A7C15 * (epoch + 1) + index) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    value ^= value >> 31
    return bool(value & 1)


def read_binary_labels(split: Path, sample_ids: list[str]) -> np.ndarray:
    labels_path = split / "labels.csv"
    labels = pd.read_csv(labels_path, dtype={"id": str})
    if "id" not in labels.columns or len(labels.columns) != 2:
        raise ValueError(f"{labels_path} must contain id and one binary label column")
    label_column = next(column for column in labels.columns if column != "id")
    if label_column not in {"label", "numeric_label"}:
        raise ValueError(f"Unsupported label column {label_column!r} in {labels_path}")
    if labels["id"].isna().any() or labels["id"].duplicated().any():
        raise ValueError(f"IDs in {labels_path} must be unique and nonmissing")
    values = pd.to_numeric(labels[label_column], errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(f"{label_column} in {labels_path} must contain only 0 and 1")
    mapping = dict(zip(labels["id"], values.astype(np.float32)))
    if set(mapping) != set(sample_ids):
        raise ValueError(f"Sample and label ID sets differ in {split}")
    return np.asarray([mapping[sample_id] for sample_id in sample_ids], dtype=np.float32)


class PairDataset(Dataset):
    def __init__(
        self,
        split: Path,
        representation: Any,
        metadata: Any,
        rc_mode: str,
        seed: int,
    ) -> None:
        self.representation = representation
        self.metadata = metadata
        self.frame = representation.read_samples(split / "input")
        self.labels = read_binary_labels(split, self.frame["id"].astype(str).tolist())
        self.rc_mode = rc_mode
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        sequence, structure = row.sequence_rna, row.structure
        use_rc = self.rc_mode == "always" or (
            self.rc_mode == "augment"
            and deterministic_reverse_complement(self.seed, self.epoch, index)
        )
        if use_rc:
            sequence, structure = self.representation.reverse_complement_raw(
                sequence, structure
            )
        channels, partners = self.representation.encode_representation(
            sequence, structure, self.metadata.length
        )
        mfe = (float(row.mfe) - self.metadata.mfe_mean) / self.metadata.mfe_std
        return {
            "channels": torch.from_numpy(channels),
            "partner_indices": torch.from_numpy(partners),
            "mfe": torch.tensor([mfe], dtype=torch.float32),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class SpeciesDataset(Dataset):
    def __init__(
        self,
        split: Path,
        representation: Any,
        metadata: Any,
        rc_mode: str,
        seed: int,
    ) -> None:
        self.representation = representation
        self.metadata = metadata
        self.frame = representation.read_samples(split / "input")
        self.labels = read_binary_labels(split, self.frame["id"].astype(str).tolist())
        self.rc_mode = rc_mode
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        use_rc = self.rc_mode == "always" or (
            self.rc_mode == "augment"
            and deterministic_reverse_complement(self.seed, self.epoch, index)
        )
        encoded = self.representation.encode_sample(
            row.sequence_rna,
            row.structure,
            row.mfe,
            row.species,
            self.metadata,
            reverse_complement=use_rc,
        )
        item = {key: torch.from_numpy(value) for key, value in encoded.items()}
        item["label"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    workers: int,
) -> DataLoader:
    if batch_size < 1 or workers < 0:
        raise ValueError("Batch size must be positive and workers must be nonnegative")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )


def update_ema(ema_model: nn.Module, model: nn.Module) -> None:
    with torch.no_grad():
        source = model.state_dict()
        for key, value in ema_model.state_dict().items():
            source_value = source[key].detach()
            if value.is_floating_point():
                value.mul_(EMA_DECAY).add_(source_value, alpha=1.0 - EMA_DECAY)
            else:
                value.copy_(source_value)


def move_inputs(
    batch: dict[str, torch.Tensor], keys: tuple[str, ...], device: torch.device
) -> tuple[list[torch.Tensor], torch.Tensor]:
    non_blocking = device.type == "cuda"
    inputs = [batch[key].to(device, non_blocking=non_blocking) for key in keys]
    labels = batch["label"].to(device, non_blocking=non_blocking)
    return inputs, labels


def predict(
    model: nn.Module,
    loader: DataLoader,
    keys: tuple[str, ...],
    device: torch.device,
    criterion: nn.Module,
) -> tuple[np.ndarray, np.ndarray, float]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    loss_sum = 0.0
    sample_count = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs, label = move_inputs(batch, keys, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(*inputs)
                loss = criterion(logits, label)
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
            labels.append(label.cpu().numpy())
            loss_sum += float(loss) * len(label)
            sample_count += len(label)
    return (
        np.concatenate(labels).astype(np.int64),
        np.concatenate(probabilities).astype(np.float64),
        loss_sum / sample_count,
    )


def train_pair_component(
    train_split: Path,
    validation_split: Path,
    representation: Any,
    architecture: Any,
    metadata: Any,
    device: torch.device,
    batch_size: int,
    workers: int,
    max_epochs: int,
) -> tuple[dict[str, torch.Tensor], Any, dict[str, Any]]:
    seed_everything(PAIR_SEED, device)
    train_data = PairDataset(
        train_split, representation, metadata, "augment", PAIR_SEED
    )
    validation_data = PairDataset(
        validation_split, representation, metadata, "never", PAIR_SEED
    )
    validation_rc = PairDataset(
        validation_split, representation, metadata, "always", PAIR_SEED
    )
    train_loader = make_loader(
        train_data, batch_size, True, PAIR_SEED, device, workers
    )
    validation_loader = make_loader(
        validation_data, batch_size, False, PAIR_SEED + 1, device, workers
    )
    validation_rc_loader = make_loader(
        validation_rc, batch_size, False, PAIR_SEED + 2, device, workers
    )

    config = architecture.ModelConfig()
    model = architecture.PairMessageCNN(config).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    keys = ("channels", "partner_indices", "mfe")
    best_state = None
    best_ap = -math.inf
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_data.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            inputs, labels = move_inputs(batch, keys, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                loss = criterion(model(*inputs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model)
            loss_sum += float(loss.detach()) * len(labels)
            sample_count += len(labels)

        labels, plain_probability, plain_loss = predict(
            ema_model, validation_loader, keys, device, criterion
        )
        labels_rc, rc_probability, rc_loss = predict(
            ema_model, validation_rc_loader, keys, device, criterion
        )
        if not np.array_equal(labels, labels_rc):
            raise RuntimeError("Pair validation views have different label order")
        validation_ap = float(
            average_precision_score(labels, 0.5 * (plain_probability + rc_probability))
        )
        scheduler.step(validation_ap)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / sample_count,
                "validation_loss": 0.5 * (plain_loss + rc_loss),
                "validation_auprc": validation_ap,
            }
        )
        print(
            f"pair epoch={epoch} train_loss={loss_sum / sample_count:.6f} "
            f"validation_auprc={validation_ap:.6f}",
            flush=True,
        )
        if validation_ap > best_ap:
            best_ap = validation_ap
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in ema_model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("PairMessageCNN training produced no checkpoint")
    return best_state, config, {
        "seed": PAIR_SEED,
        "best_epoch": best_epoch,
        "best_validation_auprc": best_ap,
        "history": history,
    }


def train_species_component(
    name: str,
    seed: int,
    use_rc: bool,
    train_split: Path,
    validation_split: Path,
    representation: Any,
    architecture: Any,
    metadata: Any,
    device: torch.device,
    batch_size: int,
    workers: int,
    max_epochs: int,
) -> tuple[dict[str, torch.Tensor], Any, dict[str, Any]]:
    seed_everything(seed, device)
    train_data = SpeciesDataset(
        train_split, representation, metadata, "augment" if use_rc else "never", seed
    )
    validation_data = SpeciesDataset(
        validation_split, representation, metadata, "never", seed
    )
    validation_rc = (
        SpeciesDataset(validation_split, representation, metadata, "always", seed)
        if use_rc
        else None
    )
    train_loader = make_loader(train_data, batch_size, True, seed, device, workers)
    validation_loader = make_loader(
        validation_data, batch_size, False, seed + 1, device, workers
    )
    validation_rc_loader = (
        make_loader(validation_rc, batch_size, False, seed + 2, device, workers)
        if validation_rc is not None
        else None
    )

    config = architecture.ModelConfig(num_species=len(metadata.species_vocabulary))
    model = architecture.SpeciesGraphGRU(config).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    keys = ("channels", "partner_indices", "global_features", "species_index")
    best_state = None
    best_ap = -math.inf
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_data.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            inputs, labels = move_inputs(batch, keys, device)
            inputs[3] = inputs[3].masked_fill(
                torch.rand(inputs[3].shape, device=device) < 0.15, 0
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                loss = criterion(model(*inputs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model)
            loss_sum += float(loss.detach()) * len(labels)
            sample_count += len(labels)

        labels, probability, validation_loss = predict(
            ema_model, validation_loader, keys, device, criterion
        )
        if validation_rc_loader is not None:
            labels_rc, probability_rc, validation_loss_rc = predict(
                ema_model, validation_rc_loader, keys, device, criterion
            )
            if not np.array_equal(labels, labels_rc):
                raise RuntimeError(f"{name} validation views have different label order")
            probability = 0.5 * (probability + probability_rc)
            validation_loss = 0.5 * (validation_loss + validation_loss_rc)
        validation_ap = float(average_precision_score(labels, probability))
        scheduler.step(validation_ap)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / sample_count,
                "validation_loss": validation_loss,
                "validation_auprc": validation_ap,
            }
        )
        print(
            f"{name} epoch={epoch} train_loss={loss_sum / sample_count:.6f} "
            f"validation_auprc={validation_ap:.6f}",
            flush=True,
        )
        if validation_ap > best_ap:
            best_ap = validation_ap
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in ema_model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"{name} training produced no checkpoint")
    return best_state, config, {
        "seed": seed,
        "reverse_complement_augmentation": use_rc,
        "best_epoch": best_epoch,
        "best_validation_auprc": best_ap,
        "history": history,
    }


def prepare_output(path: Path, source: Path) -> None:
    if path.resolve() == source.resolve():
        raise ValueError("artifacts-dir cannot overwrite source-artifacts-dir")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty artifacts directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def copy_runtime_sources(source: Path, destination: Path) -> None:
    for name in (
        "data_representation.py",
        "model.py",
        "pair_message_model.py",
        "pair_message_representation.py",
    ):
        shutil.copy2(source / name, destination / name)


def main() -> None:
    args = parse_args()
    if args.max_epochs < 1:
        raise ValueError("max-epochs must be positive")
    source = args.source_artifacts_dir.resolve()
    prepare_output(args.artifacts_dir, source)
    device = resolve_device(args.device)

    species_representation = load_module(
        "premir_train_species_representation", source / "data_representation.py"
    )
    species_architecture = load_module(
        "premir_train_species_model", source / "model.py"
    )
    pair_representation = load_module(
        "premir_train_pair_representation", source / "pair_message_representation.py"
    )
    pair_architecture = load_module(
        "premir_train_pair_model", source / "pair_message_model.py"
    )

    species_metadata = species_representation.RepresentationMetadata.fit(
        args.train_data / "input"
    )
    pair_metadata = pair_representation.RepresentationMetadata.fit(
        args.train_data / "input"
    )
    species_metadata.save(args.artifacts_dir / "representation_metadata.json")
    pair_metadata.save(args.artifacts_dir / "pair_message_representation_metadata.json")

    pair_state, pair_config, pair_summary = train_pair_component(
        args.train_data,
        args.validation_data,
        pair_representation,
        pair_architecture,
        pair_metadata,
        device,
        args.pair_batch_size,
        args.workers,
        args.max_epochs,
    )
    species_rc_state, species_config, species_rc_summary = train_species_component(
        "species_rc",
        SPECIES_RC_SEED,
        True,
        args.train_data,
        args.validation_data,
        species_representation,
        species_architecture,
        species_metadata,
        device,
        args.species_batch_size,
        args.workers,
        args.max_epochs,
    )
    species_state, _, species_summary = train_species_component(
        "species",
        SPECIES_SEED,
        False,
        args.train_data,
        args.validation_data,
        species_representation,
        species_architecture,
        species_metadata,
        device,
        args.species_batch_size,
        args.workers,
        args.max_epochs,
    )

    package = {
        "format_version": 3,
        "architecture": "PreMiRNAFixedDeployment",
        "pair_message_model_config": pair_config.to_dict(),
        "pair_message_state_dicts": [pair_state],
        "species_graph_config": species_config.to_dict(),
        "species_graph_state_dicts": [species_rc_state, species_state],
    }
    model_path = args.artifacts_dir / "model.pt"
    torch.save(package, model_path)
    copy_runtime_sources(source, args.artifacts_dir)
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    components = [
        {
            "name": "pair_rc_tta_seed8675309",
            "architecture": "PairMessageCNN",
            "weight": 1 / 3,
            "rc_tta": True,
            "state_index": 0,
        },
        {
            "name": "species_rc_augmented_seed57721",
            "architecture": "SpeciesGraphGRU",
            "weight": 1 / 3,
            "rc_tta": True,
            "state_index": 0,
        },
        {
            "name": "species_new_seed271828",
            "architecture": "SpeciesGraphGRU",
            "weight": 1 / 3,
            "rc_tta": False,
            "state_index": 1,
        },
    ]
    manifest = {
        "format_version": 3,
        "model_file": "model.pt",
        "model_sha256": model_sha256,
        "selected_candidate": "fixed_three_component_retraining",
        "aggregation": "weighted_arithmetic_sigmoid_probability",
        "components": components,
        "classification_threshold": 0.5,
    }
    (args.artifacts_dir / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    training_summary = {
        "device": str(device),
        "training_rows": len(species_representation.read_samples(args.train_data / "input")),
        "validation_rows": len(
            species_representation.read_samples(args.validation_data / "input")
        ),
        "pair_rc_tta_seed8675309": pair_summary,
        "species_rc_augmented_seed57721": species_rc_summary,
        "species_new_seed271828": species_summary,
    }
    (args.artifacts_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(training_summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
