#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explicitly rebuild the saved collapsed balanced chr14 source dataset from the folding and negative-sampling pipeline."
    )
    parser.add_argument("--genome", default="benchmark/data/chr14.fa")
    parser.add_argument("--truth-bed", default="benchmark/data/hsa-precursors-no-v2.bed")
    parser.add_argument("--dataset-output", default="benchmark/balanced_benchmark/datasets/balanced_benchmark.csv")
    parser.add_argument("--work-dir", default="benchmark/tmp/balanced_benchmark_rebuild")
    parser.add_argument("--chr", default="chr14")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--max-repeat-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nonoverlap-seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    dataset_output = (repo_root / args.dataset_output).resolve()
    work_dir = (repo_root / args.work_dir).resolve()
    fold_dir = work_dir / "fold_output"
    intermediate_dir = work_dir / "intermediate"
    plots_dir = work_dir / "plots"

    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    genome = str((repo_root / args.genome).resolve())
    truth_bed = str((repo_root / args.truth_bed).resolve())

    positives = work_dir / "positives.csv"
    positives_collapsed = work_dir / "positives_collapsed.csv"
    summary = work_dir / "summary.txt"
    mirna_plot = work_dir / "mirna_mfe.png"

    python_exe = sys.executable

    run([
        python_exe,
        str((repo_root / "benchmark/fold/run_folding.py").resolve()),
        "--input", genome,
        "--output", str(fold_dir),
        "--window", str(args.window),
        "--step", str(args.step),
        "--chr", args.chr,
        "--cpus", str(args.cpus),
        "--max_repeat_frac", str(args.max_repeat_frac),
    ])

    run([
        python_exe,
        str((repo_root / "benchmark/fold/find_mirna_windows.py").resolve()),
        "--csv", str(fold_dir / "results.csv"),
        "--bed", truth_bed,
        "--output", str(positives),
        "--output_collapsed", str(positives_collapsed),
        "--summary", str(summary),
        "--plot", str(mirna_plot),
    ])

    run([
        python_exe,
        str((repo_root / "benchmark/make_negative_set/sample_negatives.py").resolve()),
        "--positives", str(positives),
        "--positives_collapsed", str(positives_collapsed),
        "--all_windows", str(fold_dir / "results.csv"),
        "--balanced", str(intermediate_dir / "balanced.csv"),
        "--imbalanced", str(intermediate_dir / "imbalanced.csv"),
        "--balanced_collapsed", str(dataset_output),
        "--imbalanced_collapsed", str(intermediate_dir / "imbalanced_collapsed.csv"),
        "--plot_dir", str(plots_dir),
        "--seed", str(args.seed),
        "--nonoverlap_seed", str(args.nonoverlap_seed),
    ])

    print(f"balanced benchmark source dataset: {dataset_output}")
    print(f"rebuild work dir: {work_dir}")


if __name__ == "__main__":
    main()
