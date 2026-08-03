from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluate import calculate_metrics, read_labels
from inference import (
    DEFAULT_ARTIFACTS,
    RUNTIME_ARTIFACTS,
    load_manifest,
    run_inference,
    sha256,
)


def test_released_manifest_matches_checkpoint() -> None:
    manifest = load_manifest(DEFAULT_ARTIFACTS)
    assert manifest["model_sha256"] == sha256(DEFAULT_ARTIFACTS / "model.pt")
    assert set(manifest["runtime_artifact_sha256"]) == set(RUNTIME_ARTIFACTS)
    for name in RUNTIME_ARTIFACTS:
        assert manifest["runtime_artifact_sha256"][name] == sha256(
            DEFAULT_ARTIFACTS / name
        )
    assert len(manifest["components"]) == 3
    assert np.isclose(sum(item["weight"] for item in manifest["components"]), 1.0)


def test_manifest_rejects_modified_runtime_artifact(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    shutil.copytree(DEFAULT_ARTIFACTS, artifacts)
    metadata_path = artifacts / "representation_metadata.json"
    metadata_path.write_text(metadata_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="representation_metadata.json"):
        load_manifest(artifacts)


def test_inference_preserves_ids_and_probability_contract(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    samples = pd.DataFrame(
        {
            "id": ["sample_1", "sample_2"],
            "species": ["hsa", "unknown_species"],
            "sequence_rna": ["A" * 200, "C" * 200],
            "structure": ["." * 200, "." * 200],
            "mfe": [-20.0, -25.0],
        }
    )
    samples.to_csv(input_dir / "samples.tsv", sep="\t", index=False)
    output = tmp_path / "predictions.csv"
    result = run_inference(input_dir, output, requested_device="cpu", workers=0)
    assert result["id"].tolist() == ["sample_1", "sample_2"]
    assert result.columns.tolist() == [
        "id",
        "prediction",
        "probability_0",
        "probability_1",
    ]
    assert np.allclose(result["probability_0"] + result["probability_1"], 1.0)
    assert output.is_file()


def test_metrics_and_public_label_column(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text("id,label\na,0\nb,1\nc,1\n")
    labels = read_labels(labels_path)
    predictions = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "prediction": [0, 1, 0],
            "probability_0": [0.9, 0.2, 0.6],
            "probability_1": [0.1, 0.8, 0.4],
        }
    )
    metrics = calculate_metrics(predictions, labels)
    assert metrics["n"] == 3
    assert metrics["TN"] == 1
    assert metrics["TP"] == 1
    assert metrics["FN"] == 1
    assert 0.0 <= metrics["AP"] <= 1.0


def test_manifest_is_valid_json() -> None:
    manifest_path = DEFAULT_ARTIFACTS / "deployment_manifest.json"
    assert json.loads(manifest_path.read_text())["format_version"] == 3
