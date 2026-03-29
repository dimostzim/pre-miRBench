#!/usr/bin/env python3
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = ("deepmir", "deepmirgene", "dnnpremir", "mirdnn", "mire2e", "mustard")
WINDOW_TOOLS = {
    "deepmir": 200,
    "deepmirgene": 200,
    "dnnpremir": 180,
    "mirdnn": 160,
}
RESULT_FILES = {
    "deepmir": "results.csv",
    "deepmirgene": "predictions.txt",
    "dnnpremir": "predictions.txt",
    "mirdnn": "predictions.csv",
    "mire2e": "predictions.json",
}


def run(cmd, cwd):
    print("+", " ".join(str(part) for part in cmd))
    subprocess.check_call(cmd, cwd=cwd)


def load_metrics(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["tool"]: row for row in rows}


def load_truth_names(path, chrom):
    names = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 4 and parts[0] == chrom:
                names.append(parts[3])
    return names


def load_chrom_sequence(path, chrom):
    sequence_parts = []
    active = False
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:].split()[0]
                if active:
                    break
                active = header == chrom
                continue
            if active:
                sequence_parts.append(line.upper().replace("T", "U"))
    if not sequence_parts:
        raise FileNotFoundError(f"Unable to find chromosome '{chrom}' in {path}")
    return "".join(sequence_parts)


def count_valid_external_windows(sequence, window_length, stride):
    valid = 0
    for start in range(0, len(sequence) - window_length + 1, stride):
        window = sequence[start:start + window_length]
        if set(window) <= set("ACGU"):
            valid += 1
    return valid * 2


def check_balanced_dataset(repo_root, chrom):
    dataset_path = repo_root / "benchmark" / "balanced_benchmark" / "datasets" / "balanced_benchmark.csv"
    truth_path = repo_root / "benchmark" / "data" / "hsa-precursors-no-v2.bed"

    with dataset_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {dataset_path}")

    positives = [row for row in rows if row["label"] == "positive"]
    negatives = [row for row in rows if row["label"] == "negative"]
    unique_targets = {row["target_mirna"] for row in positives if row["target_mirna"]}
    sequence_lengths = {len(row["sequence"]) for row in rows}

    if len(positives) != len(negatives):
        raise RuntimeError(
            f"Balanced dataset is not 1:1: {len(positives)} positives vs {len(negatives)} negatives"
        )
    if len(unique_targets) != len(positives):
        raise RuntimeError(
            f"Collapsed positives are not one-per-target: {len(unique_targets)} unique targets vs {len(positives)} rows"
        )
    if sequence_lengths != {200}:
        raise RuntimeError(f"Expected all balanced source sequences to be 200 nt, found lengths: {sorted(sequence_lengths)}")

    truth_names = load_truth_names(truth_path, chrom=chrom)
    missing_truth = sorted(name for name in truth_names if name not in unique_targets)

    print(
        f"balanced dataset: {len(positives)} positives, {len(negatives)} negatives, "
        f"{len(truth_names)} truth loci on {chrom}, {len(missing_truth)} truth loci absent after source-window filtering"
    )
    if missing_truth:
        print("missing truth loci:", ", ".join(missing_truth))


def check_mustard_conservation_asset(repo_root, chrom):
    wigfix_path = repo_root / "benchmark" / "data" / f"{chrom}.wigFix.gz"
    if not wigfix_path.exists():
        raise FileNotFoundError(f"Missing MuStARD conservation asset: {wigfix_path}")
    print(f"MuStARD conservation asset: found {wigfix_path.name}")


def balanced_results_present(repo_root, prefix):
    missing = []
    results_root = repo_root / "results"
    for tool in TOOLS:
        tool_dir = results_root / tool / prefix
        if tool == "mustard":
            matches = sorted(tool_dir.glob("predict/static/results/intermediate_files/*.predictions.txt.gz"))
            if not matches:
                missing.append(f"{tool}:{tool_dir}")
            continue
        result_path = tool_dir / RESULT_FILES[tool]
        if not result_path.exists():
            missing.append(f"{tool}:{result_path}")
    return missing


def check_balanced_legacy_compat(repo_root, prefix):
    saved_metrics = repo_root / "benchmark" / "evaluated" / prefix / "metrics.csv"
    if not saved_metrics.exists():
        print(f"skip balanced legacy compatibility: missing saved metrics at {saved_metrics}")
        return

    missing_results = balanced_results_present(repo_root, prefix)
    if missing_results:
        print("skip balanced legacy compatibility: missing raw results for", ", ".join(missing_results))
        return

    tmp_root = repo_root / "benchmark" / "tmp" / "validate_benchmarks"
    prepared_dir = tmp_root / "prepared"
    output_dir = tmp_root / "evaluated"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(repo_root / "benchmark" / "balanced_benchmark" / "prepare_inputs.py"),
            "--input",
            str(repo_root / "benchmark" / "balanced_benchmark" / "datasets" / "balanced_benchmark.csv"),
            "--output-dir",
            str(prepared_dir),
            "--prefix",
            prefix,
        ],
        cwd=repo_root,
    )
    run(
        [
            sys.executable,
            str(repo_root / "benchmark" / "balanced_benchmark" / "evaluate_outputs.py"),
            "--prepared-dir",
            str(prepared_dir),
            "--results-dir",
            str(repo_root / "results"),
            "--output-dir",
            str(output_dir),
            "--prefix",
            prefix,
        ],
        cwd=repo_root,
    )

    old_metrics = load_metrics(saved_metrics)
    new_metrics = load_metrics(output_dir / "metrics.csv")
    if old_metrics != new_metrics:
        raise RuntimeError("Balanced legacy metrics do not match the refactored pipeline output")

    print(f"balanced legacy compatibility: regenerated metrics match saved {prefix} metrics")


def check_scan_manifests(repo_root):
    prepared_dir = repo_root / "benchmark" / "prepared_inputs" / "scan_benchmark"
    chunks_path = prepared_dir / "chunks.csv"
    genome_path = repo_root / "benchmark" / "data" / "chr14.fa"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing scan chunk manifest: {chunks_path}")

    with chunks_path.open(newline="") as handle:
        chunk_rows = list(csv.DictReader(handle))
    if not chunk_rows:
        raise RuntimeError(f"No rows found in {chunks_path}")

    previous_core_end = 0
    chrom_size_bp = 0
    chunk_ids = set()
    for row in chunk_rows:
        core_start = int(row["core_start"])
        core_end = int(row["core_end"])
        native_start = int(row["native_start"])
        native_end = int(row["native_end"])
        if core_start != previous_core_end + 1:
            raise RuntimeError(
                f"Scan chunk core coverage is not contiguous at {row['chunk_id']}: "
                f"expected {previous_core_end + 1}, found {core_start}"
            )
        if native_start > core_start or native_end < core_end:
            raise RuntimeError(f"Invalid native bounds for {row['chunk_id']}")
        previous_core_end = core_end
        chrom_size_bp = max(chrom_size_bp, core_end)
        chunk_ids.add(row["chunk_id"])

    chrom_sequence = load_chrom_sequence(genome_path, chrom=chunk_rows[0]["chrom"])
    for tool, window_length in WINDOW_TOOLS.items():
        manifest_path = prepared_dir / tool / "scan_manifest.csv"
        with manifest_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        seen_chunk_ids = set()
        for row in rows:
            chunk_id = row["chunk_id"]
            if chunk_id not in chunk_ids:
                raise RuntimeError(f"{tool} scan manifest references unknown chunk_id: {chunk_id}")
            if chunk_id in seen_chunk_ids:
                raise RuntimeError(f"{tool} scan manifest contains duplicate chunk_id: {chunk_id}")
            seen_chunk_ids.add(chunk_id)
        total_windows = sum(int(row["window_count"]) for row in rows)
        expected_total = count_valid_external_windows(chrom_sequence, window_length, stride=50)
        if total_windows != expected_total:
            raise RuntimeError(
                f"{tool} scan manifest window total mismatch: expected {expected_total}, found {total_windows}"
            )

    for tool in ("mire2e", "mustard"):
        manifest_path = prepared_dir / tool / "scan_manifest.csv"
        with manifest_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(chunk_rows):
            raise RuntimeError(
                f"{tool} native scan manifest has {len(rows)} rows but chunks.csv has {len(chunk_rows)} rows"
            )

    print(
        f"scan manifests: contiguous {chrom_size_bp} bp core coverage with "
        f"{len(chunk_rows)} chunks; external-window totals match expected full-chromosome coverage"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Validate prepared benchmark artifacts and legacy compatibility checks.")
    parser.add_argument("--chrom", default="chr14", help="Chromosome name to use for truth-locus summary checks.")
    parser.add_argument(
        "--legacy-prefix",
        default="1_1_collapsed",
        help="Legacy balanced benchmark prefix to compare against saved metrics when results are available.",
    )
    parser.add_argument("--skip-balanced-legacy", action="store_true", help="Skip the saved legacy metric compatibility check.")
    parser.add_argument("--skip-scan", action="store_true", help="Skip scan manifest validation.")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    check_balanced_dataset(repo_root, chrom=args.chrom)
    check_mustard_conservation_asset(repo_root, chrom=args.chrom)
    if not args.skip_balanced_legacy:
        check_balanced_legacy_compat(repo_root, prefix=args.legacy_prefix)
    if not args.skip_scan:
        check_scan_manifests(repo_root)
    print("validation: ok")


if __name__ == "__main__":
    main()
