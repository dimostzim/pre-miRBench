#!/usr/bin/env python3
"""Manifest-driven inference for the pre-miRBench v2 model."""
from __future__ import annotations

import os

# Same cuBLAS setting as the Agentomics scripts; it must precede the torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xgboost as xgb


REQUIRED_OUTPUT_COLUMNS = ["id", "prediction", "probability_0", "probability_1"]
TREE_FEATURES = 16354
TREE_THREADS = 24
TORCH_THREADS = 16
# Unchanged Agentomics feature, representation, and architecture code.
MODEL_CODE_FILES = (
    "augmented_feature_config.json",
    "frozen_representation.py",
    "loop_features.py",
    "neural_bundle/feature_config.json",
    "neural_bundle/features.py",
    "neural_bundle/iteration4_feature_config.json",
    "neural_bundle/representation.py",
    "neural_model.py",
    "representation.py",
    "tree_bundle/feature_config.json",
    "tree_bundle/features.py",
)
# The agent's representation wrapper checks these files against
# representation_manifest.json. It also loads tree_bundle/imputer.joblib, which
# only the helper XGBoost used during training.
TREE_BUNDLE_FILES = ("feature_config.json", "features.py", "imputer.joblib")
NEURAL_BUNDLE_FILES = (
    "feature_config.json",
    "features.py",
    "global_preprocessor.joblib",
    "iteration4_feature_config.json",
    "representation.py",
)
FITTED_ARTIFACTS = (
    "augmented_tree_imputer.joblib",
    "frozen_neural.pt",
    "model.ubj",
    "neural_bundle/global_preprocessor.joblib",
    "new_neural.pt",
    "representation_manifest.json",
    "tree_bundle/imputer.joblib",
)
RUNTIME_ARTIFACTS = tuple(sorted(MODEL_CODE_FILES + FITTED_ARTIFACTS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict precursor-miRNA candidates")
    parser.add_argument("--input", required=True, help="Input directory containing samples.tsv")
    parser.add_argument("--output", required=True, help="Destination prediction CSV")
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        type=Path,
        help="Directory containing the model bundle and deployment_manifest.json",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto)",
    )
    return parser.parse_args()


def load_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(artifacts: Path) -> dict[str, Any]:
    path = artifacts / "deployment_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format_version",
        "blend_weights",
        "classification_threshold",
        "tree_rounds",
        "artifact_sha256",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")
    if manifest["format_version"] != 1:
        raise ValueError(f"Unsupported manifest format: {manifest['format_version']}")

    weights = manifest["blend_weights"]
    if not isinstance(weights, list) or len(weights) != 3:
        raise ValueError("blend_weights must list the tree, frozen_neural, and new_neural weights")
    weights = [float(weight) for weight in weights]
    if not np.isfinite(weights).all() or min(weights) < 0:
        raise ValueError("Manifest blend weights are invalid")
    if not np.isclose(sum(weights), 1.0, atol=1e-8):
        raise ValueError("Manifest blend weights must sum to one")
    threshold = float(manifest["classification_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Classification threshold must be in [0,1]")
    if not isinstance(manifest["tree_rounds"], int) or manifest["tree_rounds"] < 1:
        raise ValueError("tree_rounds must be a positive integer")

    hashes = manifest["artifact_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(RUNTIME_ARTIFACTS):
        raise ValueError("Manifest artifact hashes are incomplete")
    for name in RUNTIME_ARTIFACTS:
        artifact = artifacts / name
        if not artifact.is_file() or sha256(artifact) != hashes[name]:
            raise RuntimeError(f"Artifact SHA-256 mismatch: {name}")
    return manifest


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def configure_torch() -> None:
    # Thread and kernel settings of the Agentomics training and inference scripts.
    torch.set_num_threads(TORCH_THREADS)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def load_representation(artifacts: Path) -> Any:
    # representation.py imports frozen_representation by its module name.
    load_module("frozen_representation", artifacts / "frozen_representation.py")
    module = load_module("premir_v2_representation", artifacts / "representation.py")
    return module.Representations(artifacts)


def load_network(architecture: Any, path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    network = architecture.Classifier(checkpoint["dropout"]).to(device)
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    return network


def write_predictions(result: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        result.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def run_inference(
    input_dir: Path,
    output: Path,
    artifacts: Path,
    requested_device: str = "auto",
) -> pd.DataFrame:
    artifacts = artifacts.resolve()
    manifest = load_manifest(artifacts)
    device = resolve_device(requested_device)
    configure_torch()

    representation = load_representation(artifacts)
    architecture = load_module("premir_v2_neural_model", artifacts / "neural_model.py")
    frozen = load_network(architecture, artifacts / "frozen_neural.pt", device)
    new = load_network(architecture, artifacts / "new_neural.pt", device)
    tree = xgb.Booster()
    tree.load_model(artifacts / "model.ubj")
    tree.set_param({"device": device.type, "nthread": TREE_THREADS})
    rounds = tree.num_boosted_rounds()
    if tree.num_features() != TREE_FEATURES or rounds != manifest["tree_rounds"]:
        raise ValueError("Tree model does not match the deployment manifest")

    frame = representation.load_samples(input_dir)
    if frame.empty:
        raise ValueError("Input contains no samples")
    ids = frame["id"].tolist()

    tree_parts, frozen_parts, new_parts = [], [], []
    # Batches of at most 256 rows share one feature extraction across the three models.
    for _, (tree_inputs, global_inputs, positional_inputs) in representation.batches(frame):
        with torch.inference_mode():
            global_tensor = torch.from_numpy(global_inputs).to(device)
            positional_tensor = torch.from_numpy(positional_inputs).to(device)
            for parts, network in ((frozen_parts, frozen), (new_parts, new)):
                probability = torch.sigmoid(network(global_tensor, positional_tensor))
                parts.append(probability.cpu().numpy().reshape(-1).astype(np.float64))
        tree_parts.append(tree.predict(xgb.DMatrix(tree_inputs)).astype(np.float64))

    tree_weight, frozen_weight, new_weight = (float(w) for w in manifest["blend_weights"])
    total = (
        tree_weight * np.concatenate(tree_parts)
        + frozen_weight * np.concatenate(frozen_parts)
        + new_weight * np.concatenate(new_parts)
    )
    if len(total) != len(ids) or not np.isfinite(total).all():
        raise RuntimeError("Inference did not produce one finite prediction per input")
    if ((total < 0.0) | (total > 1.0)).any():
        raise RuntimeError("Predicted probabilities fall outside [0,1]")
    threshold = float(manifest["classification_threshold"])
    result = pd.DataFrame(
        {
            "id": ids,
            "prediction": (total >= threshold).astype(np.int64),
            "probability_0": 1.0 - total,
            "probability_1": total,
        },
        columns=REQUIRED_OUTPUT_COLUMNS,
    )
    write_predictions(result, output)
    return result


def main() -> None:
    args = parse_args()
    run_inference(
        input_dir=Path(args.input),
        output=Path(args.output),
        artifacts=args.artifacts_dir,
        requested_device=args.device,
    )


if __name__ == "__main__":
    main()
