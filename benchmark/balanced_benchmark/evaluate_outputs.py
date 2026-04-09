#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from metrics import compute_auc_metrics, compute_binary_metrics
from tool_adapters import normalize_tool_output, parse_tools, resolve_result_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate balanced benchmark tool outputs with binary and AUC metrics."
    )
    parser.add_argument("--prepared-dir", default="benchmark/prepared_inputs/balanced_benchmark")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="benchmark/evaluated/balanced_benchmark")
    parser.add_argument("--prefix", default="balanced_benchmark")
    parser.add_argument("--tools", default="all")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def write_rows(path, rows):
    fieldnames = ["record_id", "window_id", "score", "predicted_class", "ground_truth_class"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "record_id": row["record_id"],
                "window_id": row["window_id"],
                "score": "" if row["score"] is None else f"{float(row['score']):.6f}",
                "predicted_class": row["predicted_class"],
                "ground_truth_class": row["ground_truth_class"],
            })


def append_metrics(path, tool, binary_metrics, auc_metrics):
    fieldnames = [
        "tool", "n", "tp", "fp", "tn", "fn",
        "precision", "recall", "specificity", "accuracy", "f1", "mcc",
        "roc_auc", "pr_auc",
    ]
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        roc_auc = auc_metrics.get("roc_auc")
        pr_auc = auc_metrics.get("pr_auc")
        writer.writerow({
            "tool": tool,
            "n": binary_metrics["n"],
            "tp": binary_metrics["tp"],
            "fp": binary_metrics["fp"],
            "tn": binary_metrics["tn"],
            "fn": binary_metrics["fn"],
            "precision": f"{binary_metrics['precision']:.6f}",
            "recall": f"{binary_metrics['recall']:.6f}",
            "specificity": f"{binary_metrics['specificity']:.6f}",
            "accuracy": f"{binary_metrics['accuracy']:.6f}",
            "f1": f"{binary_metrics['f1']:.6f}",
            "mcc": f"{binary_metrics['mcc']:.6f}",
            "roc_auc": "" if roc_auc is None else f"{roc_auc:.6f}",
            "pr_auc": "" if pr_auc is None else f"{pr_auc:.6f}",
        })


def main():
    args = parse_args()
    prepared_dir = Path(args.prepared_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.csv"
    curves_path = output_dir / "curves.json"

    if metrics_path.exists():
        metrics_path.unlink()
    if curves_path.exists():
        curves_path.unlink()

    all_curves = {}

    for tool in parse_tools(args.tools):
        metadata_path = prepared_dir / tool / f"{args.prefix}.metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for {tool}: {metadata_path}")

        raw_output = resolve_result_path(tool, results_dir, args.prefix)
        rows = normalize_tool_output(tool, raw_output, metadata_path, threshold=args.threshold)

        normalized_path = output_dir / f"{tool}.csv"
        write_rows(normalized_path, rows)

        binary_metrics = compute_binary_metrics(rows)
        auc_metrics = compute_auc_metrics(rows)
        append_metrics(metrics_path, tool, binary_metrics, auc_metrics)

        all_curves[tool] = {
            "fpr": auc_metrics.get("fpr"),
            "tpr": auc_metrics.get("tpr"),
            "precision_curve": auc_metrics.get("precision_curve"),
            "recall_curve": auc_metrics.get("recall_curve"),
            "roc_auc": auc_metrics.get("roc_auc"),
            "pr_auc": auc_metrics.get("pr_auc"),
        }

        roc_str = f"{auc_metrics['roc_auc']:.4f}" if auc_metrics.get("roc_auc") is not None else "N/A"
        pr_str = f"{auc_metrics['pr_auc']:.4f}" if auc_metrics.get("pr_auc") is not None else "N/A"
        print(f"{tool}: ROC AUC={roc_str}  PR AUC={pr_str}  → {normalized_path}")

    with curves_path.open("w") as f:
        json.dump(all_curves, f)

    print(f"metrics: {metrics_path}")
    print(f"curves:  {curves_path}")


if __name__ == "__main__":
    main()
