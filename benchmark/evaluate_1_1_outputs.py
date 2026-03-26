#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from metrics import compute_binary_metrics
from tool_adapters import normalize_tool_output, parse_tools, resolve_result_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate collapsed 1:1 tool outputs with label-based metrics shared by every tool."
    )
    parser.add_argument("--prepared-dir", default="benchmark/prepared_inputs/1_1_collapsed", help="Directory created by prepare_1_1_inputs.py")
    parser.add_argument("--results-dir", default="results", help="Directory containing tool outputs")
    parser.add_argument("--output-dir", default="benchmark/evaluated/1_1_collapsed", help="Directory to write normalized outputs and metrics into")
    parser.add_argument("--prefix", default="1_1_collapsed", help="Dataset/output prefix, e.g. 1_1_collapsed")
    parser.add_argument("--tools", default="all", help="Comma-separated tool list or 'all'")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for score-based tools")
    parser.add_argument("--mustard-positive-column", type=int, default=0, help="MuStARD score column to treat as positive (source indicates class_0 is positive)")
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


def append_metrics(path, tool, metrics):
    fieldnames = [
        "tool", "n", "tp", "fp", "tn", "fn",
        "precision", "recall", "specificity", "accuracy", "f1", "mcc",
    ]
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
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
        })


def main():
    args = parse_args()
    prepared_dir = Path(args.prepared_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    for tool in parse_tools(args.tools):
        metadata_path = prepared_dir / tool / f"{args.prefix}.metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for {tool}: {metadata_path}")

        raw_output = resolve_result_path(tool, results_dir, args.prefix)
        rows = normalize_tool_output(
            tool,
            raw_output,
            metadata_path,
            threshold=args.threshold,
            mustard_positive_column=args.mustard_positive_column,
        )

        normalized_path = output_dir / f"{tool}.csv"
        write_rows(normalized_path, rows)
        append_metrics(metrics_path, tool, compute_binary_metrics(rows))
        print(f"{tool}: {normalized_path}")

    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
