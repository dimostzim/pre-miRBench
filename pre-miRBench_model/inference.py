#!/usr/bin/env python3
"""Manifest-driven inference for the released pre-miRBench model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = MODEL_DIR / "training_artifacts"
REQUIRED_OUTPUT_COLUMNS = ["id", "prediction", "probability_0", "probability_1"]
RUNTIME_ARTIFACTS = (
    "data_representation.py",
    "model.py",
    "pair_message_model.py",
    "pair_message_representation.py",
    "pair_message_representation_metadata.json",
    "representation_metadata.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict precursor-miRNA candidates")
    parser.add_argument("--input", required=True, help="Input directory containing samples.tsv")
    parser.add_argument("--output", required=True, help="Destination prediction CSV")
    parser.add_argument(
        "--artifacts-dir",
        default=DEFAULT_ARTIFACTS,
        type=Path,
        help="Directory containing model and representation artifacts",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Data-loader workers; zero is the portable default",
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
        "model_file",
        "model_sha256",
        "runtime_artifact_sha256",
        "aggregation",
        "components",
        "classification_threshold",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")
    if manifest["format_version"] != 3:
        raise ValueError(f"Unsupported manifest format: {manifest['format_version']}")
    if manifest["aggregation"] != "weighted_arithmetic_sigmoid_probability":
        raise ValueError("Unsupported deployment aggregation")

    components = manifest["components"]
    if not isinstance(components, list) or not 1 <= len(components) <= 3:
        raise ValueError("Deployment must contain between one and three components")
    weights = []
    for component in components:
        needed = {"name", "architecture", "weight", "rc_tta", "state_index"}
        if needed - component.keys():
            raise ValueError(f"Malformed component: {component}")
        if component["architecture"] not in {"PairMessageCNN", "SpeciesGraphGRU"}:
            raise ValueError(f"Unsupported architecture: {component['architecture']}")
        if not isinstance(component["rc_tta"], bool):
            raise ValueError("rc_tta must be boolean")
        if not isinstance(component["state_index"], int) or component["state_index"] < 0:
            raise ValueError("state_index must be a nonnegative integer")
        weights.append(float(component["weight"]))
    if not np.isfinite(weights).all() or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("Manifest component weights are invalid")
    if not np.isclose(sum(weights), 1.0, atol=1e-8):
        raise ValueError("Manifest component weights must sum to one")

    threshold = float(manifest["classification_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Classification threshold must be in [0,1]")
    model_path = artifacts / manifest["model_file"]
    if not model_path.is_file() or sha256(model_path) != manifest["model_sha256"]:
        raise RuntimeError("Model artifact SHA-256 does not match deployment manifest")

    runtime_hashes = manifest["runtime_artifact_sha256"]
    if (
        not isinstance(runtime_hashes, dict)
        or set(runtime_hashes) != set(RUNTIME_ARTIFACTS)
    ):
        raise ValueError("Manifest runtime artifact hashes are incomplete")
    for name in RUNTIME_ARTIFACTS:
        runtime_path = artifacts / name
        if not runtime_path.is_file() or sha256(runtime_path) != runtime_hashes[name]:
            raise RuntimeError(f"Runtime artifact SHA-256 mismatch: {name}")
    return manifest


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


class SpeciesViewDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, representation: Any, metadata: Any, rc: bool):
        self.frame = frame
        self.representation = representation
        self.metadata = metadata
        self.rc = rc

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        value = self.representation.encode_sample(
            row.sequence_rna,
            row.structure,
            float(row.mfe),
            row.species,
            self.metadata,
            self.rc,
        )
        return {
            "channels": torch.from_numpy(value["channels"]),
            "partner_indices": torch.from_numpy(value["partner_indices"]),
            "global_features": torch.from_numpy(value["global_features"]),
            "species_index": torch.as_tensor(value["species_index"], dtype=torch.long),
        }


class PairViewDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        pair_representation: Any,
        metadata: Any,
        rc: bool,
        species_representation: Any,
    ):
        self.frame = frame
        self.representation = pair_representation
        self.metadata = metadata
        self.rc = rc
        self.species_representation = species_representation

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        sequence, structure, mfe = row.sequence_rna, row.structure, float(row.mfe)
        if self.rc:
            sequence, structure, mfe = self.species_representation.reverse_complement_view(
                sequence, structure, mfe
            )
        channels, partners = self.representation.encode_representation(
            sequence, structure, self.metadata.length
        )
        standardized_mfe = (float(mfe) - self.metadata.mfe_mean) / self.metadata.mfe_std
        if not np.isfinite(channels).all() or not np.isfinite(standardized_mfe):
            raise ValueError(f"Non-finite pair representation at row {index}")
        return {
            "channels": torch.from_numpy(channels),
            "partner_indices": torch.from_numpy(partners),
            "mfe": torch.tensor([standardized_mfe], dtype=torch.float32),
        }


def make_loader(
    dataset: Dataset,
    device: torch.device,
    batch_size: int | None,
    workers: int,
) -> DataLoader:
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    if batch_size is None:
        batch_size = 256 if device.type == "cuda" else 64
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )


def predict_model(
    model: torch.nn.Module,
    loader: DataLoader,
    keys: tuple[str, ...],
    device: torch.device,
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = [batch[key].to(device, non_blocking=device.type == "cuda") for key in keys]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(*inputs)
            probabilities = torch.sigmoid(logits.float())
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Model generated non-finite probabilities")
            output.append(probabilities.cpu().numpy())
    if not output:
        raise ValueError("Input contains no samples")
    return np.concatenate(output).astype(np.float64, copy=False)


def safely_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        package = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        package = torch.load(path, map_location="cpu")
    if not isinstance(package, dict):
        raise ValueError("Invalid model checkpoint")
    return package


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
    artifacts: Path = DEFAULT_ARTIFACTS,
    requested_device: str = "auto",
    batch_size: int | None = None,
    workers: int = 0,
) -> pd.DataFrame:
    artifacts = artifacts.resolve()
    manifest = load_manifest(artifacts)
    species_rep = load_module(
        "premir_species_representation", artifacts / "data_representation.py"
    )
    species_arch = load_module("premir_species_model", artifacts / "model.py")
    pair_rep = load_module(
        "premir_pair_representation", artifacts / "pair_message_representation.py"
    )
    pair_arch = load_module("premir_pair_model", artifacts / "pair_message_model.py")

    species_metadata = species_rep.RepresentationMetadata.load(
        artifacts / "representation_metadata.json"
    )
    pair_metadata = pair_rep.RepresentationMetadata.load(
        artifacts / "pair_message_representation_metadata.json"
    )
    frame = species_rep.read_samples(input_dir)
    ids = frame["id"].astype(str).tolist()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Every input must have one unique, nonempty ID")

    device = resolve_device(requested_device)
    package = safely_load_checkpoint(artifacts / manifest["model_file"])
    if package.get("architecture") != "PreMiRNAFixedDeployment":
        raise ValueError("Checkpoint deployment architecture is invalid")

    species_datasets = {
        False: SpeciesViewDataset(frame, species_rep, species_metadata, False),
        True: SpeciesViewDataset(frame, species_rep, species_metadata, True),
    }
    pair_datasets = {
        False: PairViewDataset(frame, pair_rep, pair_metadata, False, species_rep),
        True: PairViewDataset(frame, pair_rep, pair_metadata, True, species_rep),
    }
    loaders: dict[tuple[str, bool], DataLoader] = {}
    total = np.zeros(len(frame), dtype=np.float64)

    for component in manifest["components"]:
        architecture = component["architecture"]
        state_index = component["state_index"]
        if architecture == "SpeciesGraphGRU":
            states = package.get("species_graph_state_dicts", [])
            if state_index >= len(states):
                raise IndexError("SpeciesGraphGRU checkpoint index is out of range")
            model = species_arch.build_model(package["species_graph_config"])
            keys = ("channels", "partner_indices", "global_features", "species_index")
            datasets = species_datasets
        else:
            states = package.get("pair_message_state_dicts", [])
            if state_index >= len(states):
                raise IndexError("PairMessageCNN checkpoint index is out of range")
            model = pair_arch.build_model(package["pair_message_model_config"])
            keys = ("channels", "partner_indices", "mfe")
            datasets = pair_datasets
        model.load_state_dict(states[state_index], strict=True)
        model.to(device)
        loader_key = (architecture, False)
        if loader_key not in loaders:
            loaders[loader_key] = make_loader(
                datasets[False], device, batch_size=batch_size, workers=workers
            )
        probability = predict_model(model, loaders[loader_key], keys, device)
        if component["rc_tta"]:
            loader_key = (architecture, True)
            if loader_key not in loaders:
                loaders[loader_key] = make_loader(
                    datasets[True], device, batch_size=batch_size, workers=workers
                )
            probability_rc = predict_model(model, loaders[loader_key], keys, device)
            probability = 0.5 * (probability + probability_rc)
        total += float(component["weight"]) * probability
        del model

    if len(total) != len(ids) or not np.isfinite(total).all():
        raise RuntimeError("Inference did not produce one finite prediction per input")
    if ((total < -1e-7) | (total > 1.0 + 1e-7)).any():
        raise RuntimeError("Predicted probabilities fall outside [0,1]")
    total = np.clip(total, 0.0, 1.0)
    probability_0 = 1.0 - total
    if not np.allclose(probability_0 + total, 1.0, rtol=0.0, atol=1e-7):
        raise RuntimeError("Output class probabilities do not sum to one")
    threshold = float(manifest["classification_threshold"])
    result = pd.DataFrame(
        {
            "id": ids,
            "prediction": (total >= threshold).astype(np.int64),
            "probability_0": probability_0,
            "probability_1": total,
        },
        columns=REQUIRED_OUTPUT_COLUMNS,
    )
    if result["id"].tolist() != ids:
        raise RuntimeError("Output order does not match the input")
    write_predictions(result, output)
    return result


def main() -> None:
    args = parse_args()
    run_inference(
        input_dir=Path(args.input),
        output=Path(args.output),
        artifacts=args.artifacts_dir,
        requested_device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
