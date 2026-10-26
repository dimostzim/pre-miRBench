#!/usr/bin/env python3
"""Package the original Agentomics iteration-19 artifacts as a v2 model bundle.

The source is the agent's model_training/training_artifacts directory. Every
copied file is checked against the agent's own manifest.json before the
deployment manifest for inference.py is written.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from inference import RUNTIME_ARTIFACTS, sha256
from train import prepare_output, write_manifests


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
    agent_manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    agent_hashes = agent_manifest["files_sha256"]
    prepare_output(args.artifacts_dir)
    # The representation manifest is rewritten for the files of the v2 bundle.
    for name in RUNTIME_ARTIFACTS:
        if name == "representation_manifest.json":
            continue
        if sha256(args.source / name) != agent_hashes[name]:
            raise RuntimeError(f"Source artifact does not match the agent manifest: {name}")
        (args.artifacts_dir / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source / name, args.artifacts_dir / name)
    write_manifests(args.artifacts_dir, agent_manifest["rounds"])


if __name__ == "__main__":
    main()
