#!/usr/bin/env python3
"""
Run the full training-data pipeline and assemble tool-ready inputs.

Default output ratio is 1:5 positive:negative. The ratio is negatives per
positive, so use --ratio 1 for a balanced 1:1 dataset or --ratio 5 for 1:5.
"""
import argparse
import csv
import math
import os
import random
import subprocess
import sys
from pathlib import Path

from prepare_tool_inputs import DEFAULT_TARGET_LENGTHS, prepare_tool_inputs


FIELDNAMES = [
    "record_id",
    "split",
    "window_id",
    "chrom",
    "start",
    "end",
    "strand",
    "sequence",
    "structure",
    "mfe",
    "mirna_id",
    "target_start",
    "target_end",
    "label",
    "score",
    "consensus",
    "hard_round",
]


def run(cmd, reuse_output=None):
    if reuse_output and Path(reuse_output).exists():
        print(f"reuse existing: {reuse_output}")
        return
    print("+", " ".join(str(part) for part in cmd))
    subprocess.check_call([str(part) for part in cmd])


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_dataset(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field, "") for field in FIELDNAMES}
            writer.writerow(output)


def write_fasta(path, rows):
    with open(path, "w") as handle:
        for index, row in enumerate(rows, start=1):
            name = row.get("mirna_id") or row.get("window_id") or f"record_{index:06d}"
            handle.write(f">{name}\n")
            sequence = row["sequence"].upper().replace("T", "U")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset:offset + 80] + "\n")


def safe_bed_name(value, fallback):
    value = value or fallback
    return value.replace(" ", "_")


def write_bed(path, rows, label):
    with open(path, "w") as handle:
        for index, row in enumerate(rows, start=1):
            fallback = f"{'pos' if label == 1 else 'neg'}_{index:06d}"
            name = safe_bed_name(row.get("mirna_id") or row.get("window_id"), fallback)
            handle.write(
                "\t".join(
                    [
                        row["chrom"],
                        str(int(float(row["start"]))),
                        str(int(float(row["end"]))),
                        name,
                        str(label),
                        row.get("strand", "+"),
                    ]
                )
                + "\n"
            )


def split_counts(total, valid_frac, test_frac):
    test_count = int(round(total * test_frac))
    valid_count = int(round(total * valid_frac))
    if test_frac > 0 and total >= 3 and test_count == 0:
        test_count = 1
    if valid_frac > 0 and total >= 3 and valid_count == 0:
        valid_count = 1
    while valid_count + test_count >= total and (valid_count or test_count):
        if test_count >= valid_count and test_count:
            test_count -= 1
        elif valid_count:
            valid_count -= 1
    return valid_count, test_count


def assign_records(rows, prefix, label, valid_frac, test_frac, seed):
    rng = random.Random(seed)
    rows = [row.copy() for row in rows]
    rng.shuffle(rows)
    valid_count, test_count = split_counts(len(rows), valid_frac, test_frac)
    test_start = len(rows) - test_count
    valid_start = test_start - valid_count

    assigned = []
    for index, row in enumerate(rows, start=1):
        if index > test_start:
            split = "test"
        elif index > valid_start:
            split = "valid"
        else:
            split = "train"
        row["record_id"] = f"{prefix}_{index:06d}"
        row["split"] = split
        row["label"] = str(label)
        assigned.append(row)
    assigned.sort(key=lambda item: item["record_id"])
    return assigned


def rows_for(rows, split, label):
    return [row for row in rows if row.get("split") == split and str(row.get("label")) == str(label)]


def select_negatives(positives, hard_negatives, scored_negatives, ratio, seed):
    target = math.ceil(len(positives) * ratio)
    chosen = []
    seen = set()

    def add_rows(rows):
        for row in rows:
            if len(chosen) >= target:
                break
            key = row["window_id"]
            if key in seen:
                continue
            seen.add(key)
            chosen.append(row)

    hard_sorted = sorted(
        hard_negatives,
        key=lambda row: (
            int(row.get("hard_round") or 999999),
            -float(row.get("score") or 0.0),
        ),
    )
    scored_sorted = sorted(scored_negatives, key=lambda row: float(row.get("score") or 0.0), reverse=True)
    add_rows(hard_sorted)
    add_rows(scored_sorted)

    if len(chosen) < target:
        remaining = [row for row in scored_negatives if row["window_id"] not in seen]
        random.Random(seed).shuffle(remaining)
        add_rows(remaining)

    if len(chosen) < target:
        print(f"warning: requested {target} negatives, only {len(chosen)} available")
    return chosen


def parse_args():
    parser = argparse.ArgumentParser(description="Build trainable pre-miRNA datasets with hard negatives.")
    parser.add_argument("--genome", required=True)
    parser.add_argument("--bed", required=True, help="MirGeneDB or miRNA BED with _pre entries.")
    parser.add_argument("--work-dir", default="benchmark/train_data/work")
    parser.add_argument("--output-dir", default="data/train")
    parser.add_argument("--ratio", type=float, default=5.0, help="Negatives per positive. Default 5.0 = 1:5.")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--chr", dest="chromosomes", default=None)
    parser.add_argument("--max-repeat-frac", type=float, default=0.1)
    parser.add_argument("--min-mfe", type=float, default=-10.0)
    parser.add_argument("--min-paired-frac", type=float, default=0.40)
    parser.add_argument("--min-stem", type=int, default=8)
    parser.add_argument("--max-loop", type=int, default=25)
    parser.add_argument("--max-negative-windows", type=int, default=0, help="Candidate windows to fold. 0 = no cap.")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--mining-rounds", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=10)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--consensus", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dnnpremir-length", type=int, default=180)
    parser.add_argument("--mirdnn-length", type=int, default=160)
    parser.add_argument("--mire2e-length", type=int, default=100)
    parser.add_argument("--reuse-existing", action="store_true", help="Skip stages whose output files already exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ratio <= 0:
        raise SystemExit("--ratio must be positive")
    if args.valid_frac < 0 or args.test_frac < 0 or args.valid_frac + args.test_frac >= 1:
        raise SystemExit("--valid-frac and --test-frac must be non-negative and sum to less than 1")

    script_dir = Path(__file__).resolve().parent
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    positives_csv = work_dir / "positives.csv"
    pool_csv = work_dir / "hairpin_pool.csv"
    hard_csv = work_dir / "hard_negatives.csv"
    scores_csv = work_dir / "hard_negative_scores.csv"

    python_exe = sys.executable
    reuse = args.reuse_existing

    run(
        [
            python_exe,
            script_dir / "extract_positives.py",
            "--bed",
            args.bed,
            "--genome",
            args.genome,
            "--window",
            args.window,
            "--max-repeat-frac",
            args.max_repeat_frac,
            "--output",
            positives_csv,
            "--cpus",
            args.cpus,
        ]
        + (["--chr", args.chromosomes] if args.chromosomes else []),
        positives_csv if reuse else None,
    )

    hairpin_cmd = [
        python_exe,
        script_dir / "extract_hairpins.py",
        "--bed",
        args.bed,
        "--genome",
        args.genome,
        "--window",
        args.window,
        "--step",
        args.step,
        "--max-repeat-frac",
        args.max_repeat_frac,
        "--min-mfe",
        args.min_mfe,
        "--min-paired-frac",
        args.min_paired_frac,
        "--min-stem",
        args.min_stem,
        "--max-loop",
        args.max_loop,
        "--output",
        pool_csv,
        "--cpus",
        args.cpus,
    ]
    if args.chromosomes:
        hairpin_cmd.extend(["--chr", args.chromosomes])
    if args.max_negative_windows:
        hairpin_cmd.extend(["--max-windows", args.max_negative_windows])
    run(hairpin_cmd, pool_csv if reuse else None)

    run(
        [
            python_exe,
            script_dir / "mine_negatives.py",
            "--positives",
            positives_csv,
            "--pool",
            pool_csv,
            "--hard-negatives",
            hard_csv,
            "--scores",
            scores_csv,
            "--ratio",
            args.ratio,
            "--rounds",
            args.mining_rounds,
            "--ensemble-size",
            args.ensemble_size,
            "--trees",
            args.trees,
            "--consensus",
            args.consensus,
            "--seed",
            args.seed,
        ],
        scores_csv if reuse else None,
    )

    positives = read_rows(positives_csv)
    hard_negatives = read_rows(hard_csv)
    scored_negatives = read_rows(scores_csv)
    negatives = select_negatives(positives, hard_negatives, scored_negatives, args.ratio, args.seed)
    positives = assign_records(positives, "pos", 1, args.valid_frac, args.test_frac, args.seed)
    negatives = assign_records(negatives, "neg", 0, args.valid_frac, args.test_frac, args.seed + 1)
    dataset_rows = positives + negatives

    dataset_csv = output_dir / "dataset.csv"
    write_dataset(dataset_csv, dataset_rows)

    for split, prefix in (("train", ""), ("valid", "validation_"), ("test", "test_")):
        write_fasta(output_dir / f"{prefix}positive.fa", rows_for(dataset_rows, split, 1))
        write_fasta(output_dir / f"{prefix}negative.fa", rows_for(dataset_rows, split, 0))
        write_bed(output_dir / f"{prefix}mustard_positive.bed", rows_for(dataset_rows, split, 1), 1)
        write_bed(output_dir / f"{prefix}mustard_negative.bed", rows_for(dataset_rows, split, 0), 0)

    target_lengths = DEFAULT_TARGET_LENGTHS.copy()
    target_lengths.update(
        {
            "dnnpremir": args.dnnpremir_length,
            "mirdnn": args.mirdnn_length,
            "mire2e": args.mire2e_length,
        }
    )
    tool_output_dir = output_dir / "tool_inputs"
    prepare_tool_inputs(dataset_rows, tool_output_dir, target_lengths=target_lengths)

    print("\nsummary")
    print(f"positives: {len(positives)}")
    print(f"negatives: {len(negatives)}")
    print(f"ratio: 1:{len(negatives) / len(positives):.2f}")
    print(f"split: train={len(rows_for(dataset_rows, 'train', 1))}+{len(rows_for(dataset_rows, 'train', 0))} "
          f"valid={len(rows_for(dataset_rows, 'valid', 1))}+{len(rows_for(dataset_rows, 'valid', 0))} "
          f"test={len(rows_for(dataset_rows, 'test', 1))}+{len(rows_for(dataset_rows, 'test', 0))}")
    print(f"dataset: {dataset_csv}")
    print(f"fasta: {output_dir / 'positive.fa'} | {output_dir / 'negative.fa'}")
    print(f"bed: {output_dir / 'mustard_positive.bed'} | {output_dir / 'mustard_negative.bed'}")
    print(f"tool inputs: {tool_output_dir}")


if __name__ == "__main__":
    main()
