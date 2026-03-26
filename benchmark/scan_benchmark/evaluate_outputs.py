#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from tool_adapters import (
    evaluate_loci,
    deduplicate_window_rows,
    load_chunk_manifest,
    load_truth_rows,
    load_window_metadata,
    merge_positive_windows,
    normalize_deepmir_scan,
    normalize_deepmirgene_scan,
    normalize_dnnpremir_scan,
    normalize_mirdnn_scan,
    normalize_mire2e_scan,
    normalize_mustard_scan,
    parse_tools,
    resolve_scan_result_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the chr14 scan benchmark across all tools at the locus level."
    )
    parser.add_argument("--prepared-dir", default="benchmark/prepared_inputs/scan_benchmark", help="Directory created by benchmark/scan_benchmark/prepare_inputs.py")
    parser.add_argument("--results-dir", default="results", help="Directory containing tool outputs")
    parser.add_argument("--output-dir", default="benchmark/evaluated/scan_benchmark", help="Directory to write normalized windows, loci, and metrics into")
    parser.add_argument("--prefix", default="scan_chr14", help="Result output-name prefix used by benchmark/scan_benchmark/run_tools.sh")
    parser.add_argument("--tools", default="all", help="Comma-separated tool list or 'all'")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for score-based outputs")
    parser.add_argument("--merge-gap", type=int, default=0, help="Maximum gap allowed when merging adjacent positive windows into loci")
    parser.add_argument("--overlap-fraction", type=float, default=0.5, help="Minimum fraction of the truth locus that must be overlapped to count as a hit")
    parser.add_argument("--truth-bed", default=None, help="Optional explicit truth BED path; defaults to prepared truth_chr14.bed")
    return parser.parse_args()


def load_tool_manifest(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_window_rows(path, rows):
    fieldnames = ["chrom", "start", "end", "strand", "score", "predicted_class"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "chrom": row["chrom"],
                "start": row["start"],
                "end": row["end"],
                "strand": row["strand"],
                "score": "" if row["score"] is None else f"{float(row['score']):.6f}",
                "predicted_class": row["predicted_class"],
            })


def write_loci_rows(path, rows):
    fieldnames = ["chrom", "start", "end", "strand", "max_score", "window_count"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "chrom": row["chrom"],
                "start": row["start"],
                "end": row["end"],
                "strand": row["strand"],
                "max_score": f"{float(row['max_score']):.6f}",
                "window_count": row["window_count"],
            })


def append_metrics(path, tool, metrics):
    fieldnames = [
        "tool",
        "truth_loci",
        "predicted_loci",
        "matched_truth_loci",
        "false_positive_loci",
        "precision_locus",
        "locus_recall",
        "fp_per_mb",
    ]
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "tool": tool,
            "truth_loci": metrics["truth_loci"],
            "predicted_loci": metrics["predicted_loci"],
            "matched_truth_loci": metrics["matched_truth_loci"],
            "false_positive_loci": metrics["false_positive_loci"],
            "precision_locus": f"{metrics['precision_locus']:.6f}",
            "locus_recall": f"{metrics['locus_recall']:.6f}",
            "fp_per_mb": f"{metrics['fp_per_mb']:.6f}",
        })


def normalize_tool_rows(tool, manifest_rows, prepared_dir, results_dir, prefix, chunk_by_id, threshold):
    rows = []
    for entry in manifest_rows:
        output_path = resolve_scan_result_path(tool, results_dir, f"{prefix}/{entry['chunk_id']}")

        if tool == "deepmir":
            _, metadata_by_record = load_window_metadata(prepared_dir / entry["metadata_path"])
            rows.extend(normalize_deepmir_scan(output_path, metadata_by_record))
        elif tool == "deepmirgene":
            metadata_rows, _ = load_window_metadata(prepared_dir / entry["metadata_path"])
            rows.extend(normalize_deepmirgene_scan(output_path, metadata_rows))
        elif tool == "dnnpremir":
            _, metadata_by_record = load_window_metadata(prepared_dir / entry["metadata_path"])
            rows.extend(normalize_dnnpremir_scan(output_path, metadata_by_record))
        elif tool == "mirdnn":
            _, metadata_by_record = load_window_metadata(prepared_dir / entry["metadata_path"])
            rows.extend(normalize_mirdnn_scan(output_path, metadata_by_record, threshold))
        elif tool == "mire2e":
            rows.extend(normalize_mire2e_scan(output_path, chunk_by_id[entry["chunk_id"]], threshold))
        elif tool == "mustard":
            rows.extend(normalize_mustard_scan(output_path, threshold))
        else:
            raise RuntimeError(f"Unsupported tool: {tool}")

    return deduplicate_window_rows(rows)


def main():
    args = parse_args()
    prepared_dir = Path(args.prepared_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_rows, chunk_by_id = load_chunk_manifest(prepared_dir / "chunks.csv")
    truth_path = Path(args.truth_bed) if args.truth_bed else prepared_dir / "truth_chr14.bed"
    truth_rows = load_truth_rows(truth_path)
    chrom_size_bp = max(row["core_end"] for row in chunk_rows)

    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    for tool in parse_tools(args.tools):
        manifest_path = prepared_dir / tool / "scan_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing scan manifest for {tool}: {manifest_path}")

        manifest_rows = load_tool_manifest(manifest_path)
        window_rows = normalize_tool_rows(tool, manifest_rows, prepared_dir, results_dir, args.prefix, chunk_by_id, args.threshold)
        loci_rows = merge_positive_windows(window_rows, merge_gap=args.merge_gap)
        metrics = evaluate_loci(
            loci_rows,
            truth_rows,
            chrom_size_bp,
            overlap_fraction=args.overlap_fraction,
        )

        write_window_rows(output_dir / f"{tool}.windows.csv", window_rows)
        write_loci_rows(output_dir / f"{tool}.loci.csv", loci_rows)
        append_metrics(metrics_path, tool, metrics)
        print(f"{tool}: {output_dir / f'{tool}.windows.csv'}")
        print(f"{tool}: {output_dir / f'{tool}.loci.csv'}")

    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
