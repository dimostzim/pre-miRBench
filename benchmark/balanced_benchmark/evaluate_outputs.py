#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from metrics import compute_binary_metrics, compute_roc_auc
from tool_adapters import (
    load_metadata,
    normalize_tool_output,
    normalize_unified_output,
    parse_tools,
    resolve_result_path,
    resolve_unified_result_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate collapsed balanced benchmark tool outputs with label-based metrics shared by every tool."
    )
    parser.add_argument("--prepared-dir", default="benchmark/prepared_inputs/balanced_benchmark", help="Directory created by benchmark/balanced_benchmark/prepare_inputs.py")
    parser.add_argument("--results-dir", default="results", help="Directory containing tool outputs")
    parser.add_argument("--output-dir", default="benchmark/evaluated/balanced_benchmark", help="Directory to write normalized outputs and metrics into")
    parser.add_argument("--prefix", default="balanced_benchmark", help="Dataset/output prefix, e.g. balanced_benchmark")
    parser.add_argument("--tools", default="all", help="Comma-separated tool list or 'all'")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for score-based tools")
    parser.add_argument("--mustard-positive-column", type=int, default=1, help="MuStARD score column to treat as positive; raw predictions are written as class_0, class_1")
    parser.add_argument("--norm-output","--norm_output",choices=["y", "n"],default="n",help="When 'y', evaluate unified_predictions.csv and add threshold-free ROC-AUC to metrics.unified.csv",)
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


def append_metrics(path, tool, metrics, include_auc=False):
    fieldnames = [
        "tool", "n", "tp", "fp", "tn", "fn",
        "precision", "recall", "specificity", "accuracy", "f1", "mcc",
    ]
    if include_auc:
        fieldnames.append("auc")
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {
            "tool": tool,
            "n": metrics["n"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
            "precision": f"{metrics['precision']:.6f}",
            "recall": f"{metrics['recall']:.6f}",
            "specificity": f"{metrics['specificity']:.6f}",
            "accuracy": f"{metrics['accuracy']:.6f}",
            "f1": f"{metrics['f1']:.6f}",
            "mcc": f"{metrics['mcc']:.6f}",
        }
        if include_auc:
            row["auc"] = "" if metrics["auc"] is None else f"{metrics['auc']:.6f}"
        writer.writerow(row)


def main():
    args = parse_args()
    prepared_dir = Path(args.prepared_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / ("metrics.unified.csv" if args.norm_output == "y" else "metrics.csv")
    if metrics_path.exists():
        metrics_path.unlink()

    for tool in parse_tools(args.tools):
        metadata_path = prepared_dir / tool / f"{args.prefix}.metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for {tool}: {metadata_path}")

        if args.norm_output == "y":
            raw_output = resolve_unified_result_path(tool, results_dir, args.prefix)
            _, metadata_by_record = load_metadata(metadata_path)
            rows = normalize_unified_output(raw_output, metadata_by_record, threshold=args.threshold)
        else:
            raw_output = resolve_result_path(tool, results_dir, args.prefix)
            rows = normalize_tool_output(
                tool,
                raw_output,
                metadata_path,
                threshold=args.threshold,
                mustard_positive_column=args.mustard_positive_column,
            )

        suffix = ".unified" if args.norm_output == "y" else ""
        normalized_path = output_dir / f"{tool}{suffix}.csv"
        write_rows(normalized_path, rows)
        metrics = compute_binary_metrics(rows)
        if args.norm_output == "y":
            metrics["auc"] = compute_roc_auc(rows)
        append_metrics(metrics_path, tool, metrics, include_auc=args.norm_output == "y")
        print(f"{tool}: {normalized_path}")

    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
