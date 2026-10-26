#!/usr/bin/env python3
"""Retrain the fixed pre-miRBench v2 blend of one XGBoost and two neural networks.

The script replays the Agentomics training steps behind the iteration-19
deployment with the settings that the original seed-42 run selected, and keeps
every validation-based epoch and round selection of the original scripts:

1. extract the tree, loop, neural global, and positional features once per split;
2. fit the tree imputer (iteration 10), the augmented tree imputer (iteration 17),
   and the neural global preprocessor (iteration 11) on the training split;
3. train the helper XGBoost on the 13,357 tree features (iteration 10); it only
   selects the epochs of the two neural networks and is not deployed;
4. train the first neural network (iteration 13), selecting its epoch by the
   validation AP of 0.8 * helper XGBoost + 0.2 * first neural network;
5. train the second neural network (iteration 14), selecting its epoch by the
   validation AP of 0.8 * helper XGBoost + 0.1 * first + 0.1 * second network;
6. train the final XGBoost on the 16,354 augmented tree features (iteration 18)
   with early stopping on the validation AP of the deployed blend
   0.8 * final XGBoost + 0.1 * first + 0.1 * second network;
7. continue the final XGBoost at learning rate 0.01 (iteration 19) and keep the
   round count with the best validation AP of the deployed blend.

Seed mapping: --seed seeds the helper XGBoost, the first neural network, and
both stages of the final XGBoost; the second neural network uses --seed + 1983.
The default --seed 42 reproduces the original seeds 42 and 2025.
"""
from __future__ import annotations

import os

# Same cuBLAS setting as the Agentomics scripts; it must precede the torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import torch
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

from inference import (
    MODEL_CODE_FILES,
    NEURAL_BUNDLE_FILES,
    RUNTIME_ARTIFACTS,
    TREE_BUNDLE_FILES,
    TREE_FEATURES,
    TREE_THREADS,
    configure_torch,
    load_module,
    resolve_device,
    sha256,
)


MODEL_DIR = Path(__file__).resolve().parent
MODEL_CODE = MODEL_DIR / "model_code"
SECOND_NETWORK_SEED_OFFSET = 1983
# Deployed blend order: final XGBoost, first neural network, second neural network.
BLEND_WEIGHTS = (0.8, 0.1, 0.1)
# Epoch selection of the first neural network: helper XGBoost, first neural network.
FIRST_NETWORK_SELECTION_WEIGHTS = (0.8, 0.2)
CLASSIFICATION_THRESHOLD = 0.5
BATCH_SIZE = 256
DROPOUT = 0.4
LABEL_SMOOTHING = 0.02
MAX_EPOCHS = 100
MIN_EPOCHS = 25
EPOCH_PATIENCE = 18
FINAL_XGBOOST_ROUNDS = 16000
FINAL_XGBOOST_PATIENCE = 600
CONTINUATION_ETA = 0.01
CONTINUATION_ROUNDS = 16000
CONTINUATION_PATIENCE = 1000


class SplitFeatures(NamedTuple):
    labels: np.ndarray
    tree: np.ndarray
    loop: np.ndarray
    neural_global: np.ndarray
    positional: np.ndarray
    ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain the pre-miRBench v2 model")
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--validation-data", required=True, type=Path)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    # The Agentomics recipe trains on CUDA with fp16 autocast; cpu is for smoke tests only.
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def helper_xgboost_params(device: torch.device, seed: int) -> dict[str, Any]:
    # Iteration 10 parameters with its selected candidate (3, 2, 10).
    return dict(
        objective="binary:logistic",
        tree_method="hist",
        device=device.type,
        grow_policy="depthwise",
        max_bin=256,
        learning_rate=0.03,
        n_estimators=8000,
        subsample=0.85,
        colsample_bytree=0.85,
        colsample_bylevel=1.0,
        reg_lambda=10,
        reg_alpha=0.1,
        gamma=0.0,
        scale_pos_weight=1.0,
        random_state=seed,
        n_jobs=TREE_THREADS,
        eval_metric="aucpr",
        early_stopping_rounds=400,
        max_depth=3,
        min_child_weight=2,
    )


def final_xgboost_params(device: torch.device, seed: int) -> dict[str, Any]:
    # Iteration 18 parameters with its selected candidate (3, 2, 10).
    return dict(
        objective="binary:logistic",
        tree_method="hist",
        device=device.type,
        grow_policy="depthwise",
        max_bin=256,
        eta=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        colsample_bylevel=1.0,
        alpha=0.1,
        gamma=0,
        scale_pos_weight=1,
        seed=seed,
        nthread=TREE_THREADS,
        disable_default_eval_metric=1,
        max_depth=3,
        min_child_weight=2,
        reg_lambda=10,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_model_code() -> tuple[Any, Any, Any, Any]:
    tree_code = load_module("premir_v2_tree_features", MODEL_CODE / "tree_bundle/features.py")
    loop_code = load_module("premir_v2_loop_features", MODEL_CODE / "loop_features.py")
    neural_features = load_module(
        "premir_v2_neural_features", MODEL_CODE / "neural_bundle/features.py"
    )
    # neural_bundle/representation.py runs `import features` for the iteration-4 module.
    sys.modules["features"] = neural_features
    try:
        neural_code = load_module(
            "premir_v2_neural_representation", MODEL_CODE / "neural_bundle/representation.py"
        )
    finally:
        del sys.modules["features"]
    architecture = load_module("premir_v2_train_neural_model", MODEL_CODE / "neural_model.py")
    tree_code.validate_feature_config(MODEL_CODE / "tree_bundle/feature_config.json")
    neural_code.validate_config(MODEL_CODE / "neural_bundle")
    if len(tree_code.FEATURE_NAMES) + len(loop_code.APPENDED_NAMES) != TREE_FEATURES:
        raise ValueError("Augmented tree feature count differs from the deployment")
    return tree_code, loop_code, neural_code, architecture


def extract_split(path: Path, tree_code: Any, loop_code: Any, neural_code: Any) -> SplitFeatures:
    start = time.monotonic()
    # Input-order, one-to-one label join of iterations 10 and 18; label or numeric_label.
    frame = tree_code.load_labeled_split(path)
    loop = loop_code.extract_appended(frame)
    tree = tree_code.extract_features(frame)
    neural_global, positional = neural_code.extract_inputs(frame)
    print(
        f"extracted {path}: rows={len(frame)} seconds={time.monotonic() - start:.1f}",
        flush=True,
    )
    return SplitFeatures(
        frame["label"].to_numpy(dtype=np.int64),
        tree,
        loop,
        neural_global,
        positional,
        frame["id"].to_numpy(),
    )


def augmented_tree_inputs(split: SplitFeatures) -> np.ndarray:
    # 13,357 inherited tree features followed by 2,997 loop features (iteration 17).
    return np.concatenate((split.tree, split.loop), axis=1)


def fit_preprocessors(
    train: SplitFeatures, tree_code: Any, neural_code: Any
) -> tuple[SimpleImputer, SimpleImputer, Any]:
    tree_imputer = tree_code.make_preprocessor().fit(train.tree)
    augmented_imputer = SimpleImputer(strategy="median", keep_empty_features=True, copy=True)
    augmented_imputer.fit(augmented_tree_inputs(train))
    global_preprocessor = neural_code.make_preprocessor().fit(train.neural_global)
    return tree_imputer, augmented_imputer, global_preprocessor


def train_helper_xgboost(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.monotonic()
    params = helper_xgboost_params(device, seed)
    model = xgb.XGBClassifier(**params)
    model.fit(
        train_inputs,
        train_labels,
        eval_set=[(validation_inputs, validation_labels)],
        verbose=False,
    )
    rounds = model.best_iteration + 1
    booster = model.get_booster()[:rounds]
    probability = booster.predict(xgb.DMatrix(validation_inputs)).astype(np.float64)
    summary = {
        "parameters": params,
        "retained_rounds": rounds,
        "validation_ap": float(average_precision_score(validation_labels, probability)),
        "seconds": time.monotonic() - start,
    }
    print(
        f"helper_xgboost rounds={rounds} validation_ap={summary['validation_ap']:.6f}",
        flush=True,
    )
    return probability, summary


def neural_inputs(
    split: SplitFeatures, neural_code: Any, preprocessor: Any, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    global_inputs = neural_code.transform_global(preprocessor, split.neural_global)
    positional = torch.from_numpy(split.positional).to(device)
    return torch.from_numpy(global_inputs).to(device), positional


def predict(
    model: torch.nn.Module, global_inputs: torch.Tensor, positional: torch.Tensor
) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(global_inputs), BATCH_SIZE):
            logits = model(
                global_inputs[start:start + BATCH_SIZE], positional[start:start + BATCH_SIZE]
            )
            output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def train_network(
    name: str,
    seed: int,
    architecture: Any,
    train_inputs: tuple[torch.Tensor, torch.Tensor],
    train_labels: np.ndarray,
    validation_inputs: tuple[torch.Tensor, torch.Tensor],
    validation_labels: np.ndarray,
    fixed_blend: np.ndarray,
    network_weight: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Mixup network training of iterations 13 and 14 with epoch selection by blend AP."""
    start = time.monotonic()
    seed_everything(seed)
    global_inputs, positional = train_inputs
    model = architecture.Classifier(DROPOUT).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0003, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    targets = torch.as_tensor(
        (1 - LABEL_SMOOTHING) * train_labels + LABEL_SMOOTHING / 2, device=device
    )
    best_ap = -math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # Five warm-up epochs, then cosine decay over the remaining 95 epochs.
        lr = (
            0.00003 + (epoch - 1) * (0.0003 - 0.00003) / 4
            if epoch <= 5
            else 0.000003
            + 0.5 * (0.0003 - 0.000003) * (1 + math.cos(math.pi * (epoch - 5) / 95))
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        order = torch.randperm(len(global_inputs), device=device)
        loss_sum = 0.0
        for batch_start in range(0, len(global_inputs), BATCH_SIZE):
            index = order[batch_start:batch_start + BATCH_SIZE]
            lam = float(np.random.beta(0.2, 0.2))
            lam = max(lam, 1 - lam)
            partner = index[torch.randperm(len(index), device=device)]
            mixed_global = lam * global_inputs[index] + (1 - lam) * global_inputs[partner]
            mixed_positional = lam * positional[index] + (1 - lam) * positional[partner]
            mixed_targets = lam * targets[index] + (1 - lam) * targets[partner]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                loss = loss_fn(model(mixed_global, mixed_positional), mixed_targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} training loss is not finite")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(index)

        network_probability = predict(model, *validation_inputs)
        blend_ap = float(
            average_precision_score(
                validation_labels, fixed_blend + network_weight * network_probability
            )
        )
        if blend_ap > best_ap:
            best_ap = blend_ap
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": loss_sum / len(global_inputs),
                "network_validation_ap": float(
                    average_precision_score(validation_labels, network_probability)
                ),
                "blend_validation_ap": blend_ap,
            }
        )
        print(
            f"{name} epoch={epoch} train_loss={loss_sum / len(global_inputs):.6f} "
            f"blend_validation_ap={blend_ap:.6f}",
            flush=True,
        )
        if epoch >= MIN_EPOCHS and stale >= EPOCH_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"{name} training produced no checkpoint")
    return best_state, {
        "seed": seed,
        "label_smoothing": LABEL_SMOOTHING,
        "best_epoch": best_epoch,
        "epochs": epoch,
        "best_blend_validation_ap": best_ap,
        "seconds": time.monotonic() - start,
        "history": history,
    }


def network_from_state(
    architecture: Any, state: dict[str, torch.Tensor], device: torch.device
) -> torch.nn.Module:
    network = architecture.Classifier(DROPOUT).to(device)
    network.load_state_dict(state)
    network.eval()
    return network


def blend(tree: np.ndarray, frozen: np.ndarray, new: np.ndarray) -> np.ndarray:
    tree_weight, frozen_weight, new_weight = BLEND_WEIGHTS
    return (
        tree_weight * np.asarray(tree, dtype=np.float64)
        + frozen_weight * np.asarray(frozen, dtype=np.float64)
        + new_weight * np.asarray(new, dtype=np.float64)
    )


def blend_metric(
    labels: np.ndarray, frozen: np.ndarray, new: np.ndarray
) -> tuple[Any, dict[str, float]]:
    latest: dict[str, float] = {}

    def metric(prediction: np.ndarray, _: xgb.DMatrix) -> tuple[str, float]:
        if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
            raise RuntimeError("XGBoost produced invalid validation probabilities")
        latest["ap"] = float(average_precision_score(labels, blend(prediction, frozen, new)))
        return "blend_ap", latest["ap"]

    return metric, latest


class ContinuationSelection(xgb.callback.TrainingCallback):
    """Keep the total round count with the best validation blend AP (iteration 19)."""

    def __init__(self, initial_rounds: int, baseline_ap: float, latest: dict[str, float]):
        super().__init__()
        self.initial_rounds = initial_rounds
        self.best_ap = baseline_ap
        self.retained_rounds = initial_rounds
        self.last_improvement = 0
        self.added_rounds = 0
        self.latest = latest

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: Any) -> bool:
        total = model.num_boosted_rounds()
        self.added_rounds = total - self.initial_rounds
        if self.added_rounds != epoch + 1:
            raise RuntimeError("Continuation round count is inconsistent")
        if self.latest["ap"] > self.best_ap:
            self.best_ap = self.latest["ap"]
            self.retained_rounds = total
            self.last_improvement = self.added_rounds
        return self.added_rounds - self.last_improvement >= CONTINUATION_PATIENCE


def train_final_xgboost(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
    frozen_validation: np.ndarray,
    new_validation: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[bytearray, dict[str, Any]]:
    start = time.monotonic()
    params = final_xgboost_params(device, seed)
    metric, _ = blend_metric(validation_labels, frozen_validation, new_validation)
    train_matrix = xgb.DMatrix(train_inputs, label=train_labels)
    validation_matrix = xgb.DMatrix(validation_inputs, label=validation_labels)
    history: dict[str, Any] = {}
    stage_1 = xgb.train(
        params,
        train_matrix,
        num_boost_round=FINAL_XGBOOST_ROUNDS,
        evals=[(validation_matrix, "validation")],
        custom_metric=metric,
        maximize=True,
        early_stopping_rounds=FINAL_XGBOOST_PATIENCE,
        evals_result=history,
        verbose_eval=False,
    )
    stage_1_rounds = stage_1.best_iteration + 1
    stage_1_model = stage_1[:stage_1_rounds].save_raw("ubj")
    stage_1_seconds = time.monotonic() - start
    del stage_1, train_matrix, validation_matrix
    gc.collect()

    # Iteration 19 continued from the saved stage-1 file in a new process, with
    # new matrices; the reload keeps that starting configuration.
    start = time.monotonic()
    initial = xgb.Booster(model_file=stage_1_model)
    initial.set_param({"device": device.type, "nthread": TREE_THREADS})
    stage_1_probability = initial.predict(xgb.DMatrix(validation_inputs)).astype(np.float64)
    baseline_ap = float(
        average_precision_score(
            validation_labels, blend(stage_1_probability, frozen_validation, new_validation)
        )
    )
    continuation_params = dict(params, eta=CONTINUATION_ETA)
    metric, latest = blend_metric(validation_labels, frozen_validation, new_validation)
    selection = ContinuationSelection(stage_1_rounds, baseline_ap, latest)
    continued = xgb.train(
        continuation_params,
        xgb.DMatrix(train_inputs, label=train_labels),
        num_boost_round=CONTINUATION_ROUNDS,
        evals=[(xgb.DMatrix(validation_inputs, label=validation_labels), "validation")],
        custom_metric=metric,
        callbacks=[selection],
        xgb_model=initial,
        verbose_eval=False,
    )
    if continued.num_boosted_rounds() != stage_1_rounds + selection.added_rounds:
        raise RuntimeError("Continued booster has an unexpected round count")
    if selection.retained_rounds == stage_1_rounds:
        final_model = stage_1_model
    else:
        final_model = continued[:selection.retained_rounds].save_raw("ubj")
    summary = {
        "stage_1": {
            "parameters": params,
            "rounds": stage_1_rounds,
            "executed_rounds": len(history["validation"]["blend_ap"]),
            "validation_blend_ap": baseline_ap,
            "seconds": stage_1_seconds,
        },
        "continuation": {
            "parameters": continuation_params,
            "executed_added_rounds": selection.added_rounds,
            "added_rounds": selection.retained_rounds - stage_1_rounds,
            "retained_rounds": selection.retained_rounds,
            "validation_blend_ap": selection.best_ap,
            "seconds": time.monotonic() - start,
        },
    }
    print(
        f"final_xgboost stage_1_rounds={stage_1_rounds} "
        f"retained_rounds={selection.retained_rounds} "
        f"validation_blend_ap={selection.best_ap:.6f}",
        flush=True,
    )
    return final_model, summary


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty artifacts directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_manifests(artifacts: Path, tree_rounds: int) -> None:
    """Write the agent's representation manifest and the deployment manifest."""
    representation_manifest = {
        bundle: {"files_sha256": {name: sha256(artifacts / bundle / name) for name in names}}
        for bundle, names in (
            ("tree_bundle", TREE_BUNDLE_FILES),
            ("neural_bundle", NEURAL_BUNDLE_FILES),
        )
    }
    (artifacts / "representation_manifest.json").write_text(
        json.dumps(representation_manifest, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "format_version": 1,
        "blend_members": ["model.ubj", "frozen_neural.pt", "new_neural.pt"],
        "blend_weights": list(BLEND_WEIGHTS),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "tree_rounds": tree_rounds,
        "artifact_sha256": {name: sha256(artifacts / name) for name in RUNTIME_ARTIFACTS},
    }
    (artifacts / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    prepare_output(args.artifacts_dir)
    device = resolve_device(args.device)
    configure_torch()
    second_seed = args.seed + SECOND_NETWORK_SEED_OFFSET

    tree_code, loop_code, neural_code, architecture = load_model_code()
    train = extract_split(args.train_data, tree_code, loop_code, neural_code)
    validation = extract_split(args.validation_data, tree_code, loop_code, neural_code)
    # Split checks of iterations 18 and 19.
    if set(train.ids) & set(validation.ids):
        raise ValueError("Training and validation splits share IDs")
    for name, split in (("training", train), ("validation", validation)):
        if set(split.labels) != {0, 1}:
            raise ValueError(f"The {name} split must contain both classes")
    tree_imputer, augmented_imputer, global_preprocessor = fit_preprocessors(
        train, tree_code, neural_code
    )

    helper_validation, helper_summary = train_helper_xgboost(
        tree_imputer.transform(train.tree).astype(np.float32),
        train.labels,
        tree_imputer.transform(validation.tree).astype(np.float32),
        validation.labels,
        device,
        args.seed,
    )
    gc.collect()

    train_inputs = neural_inputs(train, neural_code, global_preprocessor, device)
    validation_inputs = neural_inputs(validation, neural_code, global_preprocessor, device)
    train_labels = train.labels.astype(np.float32)
    validation_labels = validation.labels.astype(np.float32)
    helper_weight, first_selection_weight = FIRST_NETWORK_SELECTION_WEIGHTS
    first_state, first_summary = train_network(
        "first_neural_network",
        args.seed,
        architecture,
        train_inputs,
        train_labels,
        validation_inputs,
        validation_labels,
        helper_weight * helper_validation,
        first_selection_weight,
        device,
    )
    first_validation = predict(
        network_from_state(architecture, first_state, device), *validation_inputs
    )
    tree_weight, frozen_weight, new_weight = BLEND_WEIGHTS
    second_state, second_summary = train_network(
        "second_neural_network",
        second_seed,
        architecture,
        train_inputs,
        train_labels,
        validation_inputs,
        validation_labels,
        tree_weight * helper_validation + frozen_weight * first_validation,
        new_weight,
        device,
    )
    second_validation = predict(
        network_from_state(architecture, second_state, device), *validation_inputs
    )
    del train_inputs, validation_inputs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    validation_tree_inputs = augmented_imputer.transform(
        augmented_tree_inputs(validation)
    ).astype(np.float32)
    tree_model, tree_summary = train_final_xgboost(
        augmented_imputer.transform(augmented_tree_inputs(train)).astype(np.float32),
        train.labels,
        validation_tree_inputs,
        validation.labels,
        first_validation,
        second_validation,
        device,
        args.seed,
    )

    artifacts = args.artifacts_dir
    for name in MODEL_CODE_FILES:
        (artifacts / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MODEL_CODE / name, artifacts / name)
    joblib.dump(tree_imputer, artifacts / "tree_bundle/imputer.joblib")
    joblib.dump(global_preprocessor, artifacts / "neural_bundle/global_preprocessor.joblib")
    joblib.dump(augmented_imputer, artifacts / "augmented_tree_imputer.joblib")
    for name, state, summary in (
        ("frozen_neural.pt", first_state, first_summary),
        ("new_neural.pt", second_state, second_summary),
    ):
        torch.save(
            {
                "state_dict": state,
                "dropout": DROPOUT,
                "epsilon": LABEL_SMOOTHING,
                "best_epoch": summary["best_epoch"],
            },
            artifacts / name,
        )
    (artifacts / "model.ubj").write_bytes(tree_model)
    tree_rounds = tree_summary["continuation"]["retained_rounds"]
    write_manifests(artifacts, tree_rounds)

    tree = xgb.Booster()
    tree.load_model(artifacts / "model.ubj")
    tree.set_param({"device": device.type, "nthread": TREE_THREADS})
    tree_validation = tree.predict(xgb.DMatrix(validation_tree_inputs)).astype(np.float64)
    training_history = {
        "device": str(device),
        "seeds": {
            "helper_xgboost": args.seed,
            "first_neural_network": args.seed,
            "second_neural_network": second_seed,
            "final_xgboost": args.seed,
        },
        "training_rows": len(train.labels),
        "validation_rows": len(validation.labels),
        "helper_xgboost": helper_summary,
        "first_neural_network": first_summary,
        "second_neural_network": second_summary,
        "final_xgboost": tree_summary,
        "validation_ap": {
            "final_xgboost": float(average_precision_score(validation.labels, tree_validation)),
            "first_neural_network": float(
                average_precision_score(validation.labels, first_validation)
            ),
            "second_neural_network": float(
                average_precision_score(validation.labels, second_validation)
            ),
            "blend": float(
                average_precision_score(
                    validation.labels, blend(tree_validation, first_validation, second_validation)
                )
            ),
        },
    }
    (artifacts / "training_history.json").write_text(
        json.dumps(training_history, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(training_history["validation_ap"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
