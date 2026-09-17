#!/usr/bin/env python3
"""Run the released model on the four pre-miRBench test sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from inference import DEFAULT_ARTIFACTS, run_inference


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = MODEL_DIR / "evaluation_reproduced"
TEST_SPLITS = (
    "test_known_species_known_family",
    "test_known_species_heldout_family",
    "test_heldout_species_known_family",
    "test_heldout_species_heldout_family",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the pre-miRBench model")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def read_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype={"id": str})
    if "id" not in labels.columns or len(labels.columns) != 2:
        raise ValueError(f"{path} must contain id and one binary label column")
    label_column = next(column for column in labels.columns if column != "id")
    if label_column not in {"label", "numeric_label"}:
        raise ValueError(f"Unsupported label column {label_column!r} in {path}")
    labels = labels.rename(columns={label_column: "label"})
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    if (
        labels["id"].isna().any()
        or labels["id"].duplicated().any()
        or labels["label"].isna().any()
        or not labels["label"].isin([0, 1]).all()
    ):
        raise ValueError(f"Invalid IDs or binary labels in {path}")
    labels["label"] = labels["label"].astype(np.int64)
    return labels


def calculate_metrics(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float | int]:
    merged = predictions.merge(labels, on="id", how="inner", validate="one_to_one")
    if len(merged) != len(predictions) or len(merged) != len(labels):
        raise ValueError("Prediction and label ID sets differ")
    expected_order = predictions["id"].tolist()
    if merged["id"].tolist() != expected_order:
        raise RuntimeError("Metric join changed prediction order")

    target = merged["label"].to_numpy(dtype=np.int64)
    predicted = merged["prediction"].to_numpy(dtype=np.int64)
    probability = merged["probability_1"].to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and within [0,1]")
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    return {
        "n": int(len(target)),
        "positives": int(target.sum()),
        "negatives": int(len(target) - target.sum()),
        "AP": float(average_precision_score(target, probability)),
        "AUROC": float(roc_auc_score(target, probability)),
        "ACC": float(accuracy_score(target, predicted)),
        "PRECISION": float(precision_score(target, predicted, zero_division=0)),
        "RECALL": float(recall_score(target, predicted, zero_division=0)),
        "SPECIFICITY": specificity,
        "F1": float(f1_score(target, predicted, zero_division=0)),
        "F1_MACRO": float(f1_score(target, predicted, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(target, predicted)),
        "LOG_LOSS": float(log_loss(target, probability, labels=[0, 1])),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name in TEST_SPLITS:
        split_dir = args.dataset_dir / split_name
        output_path = args.output_dir / f"eval_predictions_{split_name}.csv"
        run_inference(
            input_dir=split_dir / "input",
            output=output_path,
            artifacts=args.artifacts_dir,
            requested_device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        predictions = pd.read_csv(output_path, dtype={"id": str})
        labels = read_labels(split_dir / "labels.csv")
        metrics = calculate_metrics(predictions, labels)
        metrics_path = output_path.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        rows.append({"test_split": split_name, **metrics})
        print(f"{split_name}: AP={metrics['AP']:.12f}", flush=True)

    summary = pd.DataFrame(rows)
    mean_row = {"test_split": "mean"}
    for column in (
        "AP",
        "AUROC",
        "ACC",
        "PRECISION",
        "RECALL",
        "SPECIFICITY",
        "F1",
        "F1_MACRO",
        "MCC",
        "LOG_LOSS",
    ):
        mean_row[column] = float(summary[column].mean())
    summary = pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)
    summary.to_csv(args.output_dir / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
