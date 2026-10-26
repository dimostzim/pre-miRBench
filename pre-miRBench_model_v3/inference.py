#!/usr/bin/env python3
"""Manifest-driven inference for the pre-miRBench v3 model."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REQUIRED_OUTPUT_COLUMNS = ["id", "prediction", "probability_0", "probability_1"]
BATCH_SIZE = 256
TORCH_THREADS = 4
# Unchanged Agentomics feature, representation, and architecture code. The modules import each
# other by these top-level names, so the bundle directory is put on sys.path.
MODEL_CODE_FILES = (
    "aligned_features.py",
    "aligned_model.py",
    "alignment_schema.json",
    "features.py",
    "model.py",
    "pair_grid.py",
)
MODEL_CODE_MODULES = ("features", "pair_grid", "model", "aligned_features", "aligned_model")
# The three networks in averaging order: file, class, and input view.
COMPONENTS = (
    ("aligned_network_1.pt", "AlignedPairCNN", "aligned"),
    ("aligned_network_2.pt", "AlignedPairCNN", "aligned"),
    ("folded_grid_network.pt", "FoldedGridPreMiRNACNN", "original"),
)
FITTED_ARTIFACTS = ("preprocessing.json",) + tuple(name for name, _, _ in COMPONENTS)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(artifacts: Path) -> dict[str, Any]:
    manifest = json.loads((artifacts / "deployment_manifest.json").read_text(encoding="utf-8"))
    required = {"format_version", "components", "classification_threshold", "artifact_sha256"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")
    if manifest["format_version"] != 1:
        raise ValueError(f"Unsupported manifest format: {manifest['format_version']}")
    listed = [(c.get("file"), c.get("class"), c.get("view")) for c in manifest["components"]]
    if listed != [tuple(component) for component in COMPONENTS]:
        raise ValueError("Manifest components differ from the v3 model")
    threshold = float(manifest["classification_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Classification threshold must be in [0,1]")
    hashes = manifest["artifact_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(RUNTIME_ARTIFACTS):
        raise ValueError("Manifest artifact hashes are incomplete")
    for name in RUNTIME_ARTIFACTS:
        artifact = artifacts / name
        if not artifact.is_file() or sha256(artifact) != hashes[name]:
            raise RuntimeError(f"Artifact SHA-256 mismatch: {name}")
    return manifest


def load_model_code(directory: Path) -> dict[str, Any]:
    """Import the Agentomics modules from directory and check that the loaded files are its own."""
    directory = directory.resolve()
    sys.path.insert(0, str(directory))
    modules = {name: importlib.import_module(name) for name in MODEL_CODE_MODULES}
    for name, module in modules.items():
        if sha256(Path(module.__file__)) != sha256(directory / f"{name}.py"):
            raise RuntimeError(f"Module {name} was already loaded from another file: {module.__file__}")
    return modules


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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_network(modules: dict[str, Any], class_name: str, path: Path, device: torch.device) -> torch.nn.Module:
    module = modules["aligned_model"] if class_name == "AlignedPairCNN" else modules["model"]
    network = getattr(module, class_name)().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    return network.requires_grad_(False).eval()


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
    modules = load_model_code(artifacts)
    features = modules["features"].RNAFeatures.load(artifacts / "preprocessing.json")
    build_aligned_features = modules["aligned_features"].build_aligned_features
    networks = [load_network(modules, cls, artifacts / name, device) for name, cls, _ in COMPONENTS]

    frame = modules["features"].load_samples(input_dir)
    parts = []
    # As in the Agentomics inference: one feature extraction per batch of 256 rows, the aligned
    # view shared by both aligned networks, and the float32 mean of the three sigmoid probabilities.
    with torch.inference_mode():
        for start in range(0, len(frame), BATCH_SIZE):
            batch = frame.iloc[start:start + BATCH_SIZE]
            x, mfe, partners = [torch.from_numpy(a).to(device) for a in features.transform(batch)]
            aligned_x, aligned_partners = build_aligned_features(x, partners, batch.structure.tolist())
            logits = torch.stack([
                network(aligned_x, mfe, aligned_partners) if view == "aligned" else network(x, mfe, partners)
                for network, (_, _, view) in zip(networks, COMPONENTS)
            ])
            parts.append(logits.sigmoid().mean(0).cpu().numpy())
    probability = np.concatenate(parts).astype(np.float64)
    if len(probability) != len(frame) or not np.isfinite(probability).all():
        raise RuntimeError("Inference did not produce one finite prediction per input")
    threshold = float(manifest["classification_threshold"])
    result = pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "prediction": (probability >= threshold).astype(np.int64),
            "probability_0": 1.0 - probability,
            "probability_1": probability,
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
