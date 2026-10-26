#!/usr/bin/env python3
"""Package the original Agentomics iteration-19 artifacts as a v3 model bundle.

The source is the agent's model_training/training_artifacts directory. Every
copied file is checked against the SHA-256 hashes in the agent's
model_config.json; the checkpoints, the preprocessing and model_config.json are
also checked against provenance.json, and the model code against model_code/,
before the deployment manifest for inference.py is written. The agent's
checkpoints component_seed_42.pt, component_seed_73.pt and component_seed_17.pt
become aligned_network_1.pt, aligned_network_2.pt and folded_grid_network.pt.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from inference import COMPONENTS, MODEL_CODE_FILES, sha256
from train import DEFAULT_SEEDS, MODEL_CODE, MODEL_DIR, prepare_output, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the original iteration-19 model")
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Agentomics best_iteration_snapshot/model_training/training_artifacts directory",
    )
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_hashes = json.loads((args.source / "model_config.json").read_text(encoding="utf-8"))["sha256"]
    deployment = json.loads((MODEL_DIR / "provenance.json").read_text(encoding="utf-8"))["deployment"]
    recorded = {network["agent_file"]: network["agent_file_sha256"] for network in deployment["networks"]}
    recorded["preprocessing.json"] = deployment["preprocessing_sha256"]
    recorded["model_config.json"] = deployment["agent_model_config_sha256"]
    for name, digest in recorded.items():
        if sha256(args.source / name) != digest:
            raise RuntimeError(f"Source artifact is not the iteration-19 file recorded in provenance.json: {name}")
    copies = [(name, name) for name in MODEL_CODE_FILES + ("preprocessing.json",)]
    copies += [(f"component_seed_{seed}.pt", name) for seed, (name, _, _) in zip(DEFAULT_SEEDS, COMPONENTS)]
    for source_name, _ in copies:
        if sha256(args.source / source_name) != agent_hashes[source_name]:
            raise RuntimeError(f"Source artifact does not match the agent's model_config.json: {source_name}")
    for name in MODEL_CODE_FILES:
        if sha256(args.source / name) != sha256(MODEL_CODE / name):
            raise RuntimeError(f"Agent model code differs from model_code/: {name}")
    prepare_output(args.artifacts_dir)
    for source_name, name in copies:
        shutil.copy2(args.source / source_name, args.artifacts_dir / name)
    epochs = [torch.load(args.artifacts_dir / name, map_location="cpu", weights_only=True)["epoch"]
              for name, _, _ in COMPONENTS]
    write_manifest(args.artifacts_dir, list(DEFAULT_SEEDS), epochs)


if __name__ == "__main__":
    main()
