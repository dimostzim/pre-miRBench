#!/usr/bin/env python3
"""
Build a prefixed multi-species training dataset.

Each species genome/BED is rewritten with species-prefixed contig names before
window extraction, so chromosome names are globally unique for MuStARD and split
assignment. The default split holds out two complete species, plus one
test chromosome and one validation chromosome per remaining species when enough
positive-bearing chromosomes exist.
"""
import argparse
import csv
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from prepare_tool_inputs import DEFAULT_TARGET_LENGTHS, prepare_tool_inputs


FIELDNAMES = [
    "record_id",
    "species",
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


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if str(path).endswith(".tsv") else ","))


def read_stats(path):
    if not Path(path).exists():
        return {}
    stats = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            stats[row["metric"]] = row["value"]
    return stats


def write_csv(path, rows, fieldnames=FIELDNAMES):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def find_single_file(directory, suffix):
    matches = sorted(Path(directory).glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No *{suffix} file found in {directory}")
    if len(matches) > 1:
        raise ValueError(f"Expected one *{suffix} file in {directory}, found: {matches}")
    return matches[0]


def load_panel(path, species_filter):
    rows = read_csv(path)
    if species_filter:
        keep = set(species_filter)
        rows = [row for row in rows if row["code"] in keep]
    output = []
    for row in rows:
        if row.get("status") != "auto":
            continue
        genome = row.get("genome")
        bed = row.get("bed")
        if not genome:
            genome = str(find_single_file(Path(path).parent / row["code"], ".fa"))
        if not bed:
            bed = str(Path(path).parent / row["code"] / f"{row['code']}-precursors-no-v2.bed")
        output.append(
            {
                "code": row["code"],
                "genome": genome,
                "bed": bed,
                "bed_rows": row.get("bed_rows", ""),
                "matched_rows": row.get("matched_rows", ""),
                "dropped_rows": row.get("dropped_rows", ""),
            }
        )
    if not output:
        raise SystemExit(f"No auto-download species found in {path}")
    return output


def prefixed_name(species, name):
    return f"{species}__{name}"


def write_prefixed_fasta(species, input_path, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(input_path) as input_handle, open(output_path, "w") as output_handle:
        for raw_line in input_handle:
            if raw_line.startswith(">"):
                header = raw_line[1:].strip()
                name, *rest = header.split(maxsplit=1)
                suffix = f" {rest[0]}" if rest else ""
                output_handle.write(f">{prefixed_name(species, name)}{suffix}\n")
            else:
                output_handle.write(raw_line)


def write_prefixed_bed(species, input_path, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(input_path) as input_handle, open(output_path, "w") as output_handle:
        for raw_line in input_handle:
            if not raw_line.strip() or raw_line.startswith("#"):
                output_handle.write(raw_line)
                continue
            parts = raw_line.rstrip("\n").split("\t")
            parts[0] = prefixed_name(species, parts[0])
            output_handle.write("\t".join(parts) + "\n")


def append_file(source, destination):
    with open(source) as input_handle, open(destination, "a") as output_handle:
        for line in input_handle:
            output_handle.write(line)


def unique_rows_by_window(rows):
    unique = {}
    for row in rows:
        unique.setdefault(row["window_id"], row)
    return list(unique.values())


def choose_holdout_chroms(positives, negatives, species, heldout_species, ratio):
    if species in heldout_species:
        return "", "", []
    positive_counts = Counter(row["chrom"] for row in positives)
    negative_counts = Counter(row["chrom"] for row in unique_rows_by_window(negatives))
    ranked = [chrom for chrom, _count in sorted(positive_counts.items(), key=lambda item: (-item[1], item[0]))]
    skipped = []
    eligible = []
    for chrom in ranked:
        required = math.ceil(positive_counts[chrom] * ratio)
        available = negative_counts[chrom]
        if available >= required:
            eligible.append(chrom)
        else:
            skipped.append(
                f"{chrom}: holdout skipped, needs {required} negatives for "
                f"{positive_counts[chrom]} positives but has {available}"
            )

    issues = []
    if skipped:
        examples = "; ".join(skipped[:5])
        suffix = f"; +{len(skipped) - 5} more" if len(skipped) > 5 else ""
        issues.append(f"{len(skipped)} chromosomes/scaffolds ineligible for holdout ({examples}{suffix})")

    # Keep the largest positive-bearing chromosome in training when possible.
    train_anchor = ranked[0] if ranked else None
    holdout_candidates = [chrom for chrom in eligible if chrom != train_anchor]
    if len(holdout_candidates) >= 2:
        return holdout_candidates[0], holdout_candidates[1], issues
    if len(holdout_candidates) == 1:
        return holdout_candidates[0], "", issues
    return "", "", issues


def split_for_row(row, species, heldout_species, test_chrom, valid_chrom):
    if species in heldout_species:
        return "test_species"
    if test_chrom and row["chrom"] == test_chrom:
        return "test_chrom"
    if valid_chrom and row["chrom"] == valid_chrom:
        return "valid"
    return "train"


def group_by_split(rows, species, heldout_species, test_chrom, valid_chrom):
    grouped = defaultdict(list)
    for row in rows:
        row = row.copy()
        row["species"] = species
        row["split"] = split_for_row(row, species, heldout_species, test_chrom, valid_chrom)
        grouped[row["split"]].append(row)
    return grouped


def assign_final_records(rows, species, label, start_index):
    output = []
    for offset, row in enumerate(rows, start=start_index):
        row = row.copy()
        row["record_id"] = f"{species}_{'pos' if label == 1 else 'neg'}_{offset:06d}"
        row["species"] = species
        row["label"] = str(label)
        output.append(row)
    return output


def select_negatives_for_split(positives, hard_negatives, scored_negatives, ratio, seed):
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

    return chosen, target - len(chosen)


def write_split_summary(path, rows):
    fieldnames = [
        "species",
        "test_chrom",
        "valid_chrom",
        "bed_rows",
        "bed_matched_rows",
        "bed_dropped_rows",
        "bed_pre_entries",
        "positives_after_filters",
        "positives_written",
        "positive_chr_missing",
        "positive_boundary_filtered",
        "positive_repeat_filtered",
        "negative_windows_folded",
        "negative_hairpin_like_kept",
        "negative_overlap_skipped",
        "negative_repeat_or_n_skipped",
        "positives",
        "negatives",
        "train_pos",
        "train_neg",
        "valid_pos",
        "valid_neg",
        "test_chrom_pos",
        "test_chrom_neg",
        "test_species_pos",
        "test_species_neg",
        "issues",
    ]
    write_csv(path, rows, fieldnames=fieldnames)


def process_species(species_index, species_count, species_row, args, script_dir, python_exe, mining_jobs):
    species = species_row["code"]
    print(f"\n### species {species} ({species_index}/{species_count})")
    work_dir = Path(args.work_dir)
    species_work = work_dir / species
    species_work.mkdir(parents=True, exist_ok=True)
    prefixed_genome = species_work / "genome.prefixed.fa"
    prefixed_bed = species_work / "precursors.prefixed.bed"
    positives_csv = species_work / "positives.csv"
    positives_stats_csv = species_work / "positives.stats.csv"
    pool_csv = species_work / "hairpin_pool.csv"
    pool_stats_csv = species_work / "hairpin_pool.stats.csv"
    hard_csv = species_work / "hard_negatives.csv"
    scores_csv = species_work / "hard_negative_scores.csv"

    if not args.reuse_existing or not prefixed_genome.exists():
        write_prefixed_fasta(species, species_row["genome"], prefixed_genome)
    if not args.reuse_existing or not prefixed_bed.exists():
        write_prefixed_bed(species, species_row["bed"], prefixed_bed)

    run(
        [
            python_exe,
            script_dir / "validate_bed_genome.py",
            "--bed",
            prefixed_bed,
            "--genome",
            prefixed_genome,
        ],
        None,
    )
    run(
        [
            python_exe,
            script_dir / "extract_positives.py",
            "--bed",
            prefixed_bed,
            "--genome",
            prefixed_genome,
            "--window",
            args.window,
            "--max-repeat-frac",
            args.max_repeat_frac,
            "--output",
            positives_csv,
            "--stats-output",
            positives_stats_csv,
            "--cpus",
            args.cpus,
        ],
        positives_csv if args.reuse_existing else None,
    )

    hairpin_cmd = [
        python_exe,
        script_dir / "extract_hairpins.py",
        "--bed",
        prefixed_bed,
        "--genome",
        prefixed_genome,
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
        "--stats-output",
        pool_stats_csv,
        "--cpus",
        args.cpus,
    ]
    if args.max_negative_windows_per_species:
        hairpin_cmd.extend(["--max-windows", args.max_negative_windows_per_species])
    if not args.sequential_negative_scan:
        hairpin_cmd.append("--balance-bed-chroms")
    run(hairpin_cmd, pool_csv if args.reuse_existing else None)

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
            "--jobs",
            mining_jobs,
            "--seed",
            args.seed + species_index,
        ],
        scores_csv if args.reuse_existing else None,
    )

    heldout_species = {item.strip() for item in args.heldout_species.split(",") if item.strip()}
    positives = read_csv(positives_csv)
    positives_stats = read_stats(positives_stats_csv)
    pool_stats = read_stats(pool_stats_csv)
    hard_negatives = read_csv(hard_csv)
    scored_negatives = read_csv(scores_csv)
    negative_pool = unique_rows_by_window(hard_negatives + scored_negatives)
    test_chrom, valid_chrom, holdout_issues = choose_holdout_chroms(
        positives,
        negative_pool,
        species,
        heldout_species,
        args.ratio,
    )

    positives_by_split = group_by_split(positives, species, heldout_species, test_chrom, valid_chrom)
    hard_by_split = group_by_split(hard_negatives, species, heldout_species, test_chrom, valid_chrom)
    scored_by_split = group_by_split(scored_negatives, species, heldout_species, test_chrom, valid_chrom)

    species_rows = []
    species_issues = list(holdout_issues)
    pos_index = 1
    neg_index = 1
    for split in ("train", "valid", "test_chrom", "test_species"):
        split_positives = positives_by_split.get(split, [])
        if not split_positives:
            continue
        split_negatives, shortfall = select_negatives_for_split(
            split_positives,
            hard_by_split.get(split, []),
            scored_by_split.get(split, []),
            args.ratio,
            args.seed + species_index,
        )
        if shortfall > 0:
            requested = math.ceil(len(split_positives) * args.ratio)
            species_issues.append(
                f"{split}: requested {requested} negatives for "
                f"{len(split_positives)} positives but selected {len(split_negatives)}"
            )
        species_rows.extend(assign_final_records(split_positives, species, 1, pos_index))
        species_rows.extend(assign_final_records(split_negatives, species, 0, neg_index))
        pos_index += len(split_positives)
        neg_index += len(split_negatives)

    counts = Counter((row["split"], row["label"]) for row in species_rows)
    summary_row = {
        "species": species,
        "test_chrom": test_chrom,
        "valid_chrom": valid_chrom,
        "bed_rows": species_row.get("bed_rows", ""),
        "bed_matched_rows": species_row.get("matched_rows", ""),
        "bed_dropped_rows": species_row.get("dropped_rows", ""),
        "bed_pre_entries": positives_stats.get("bed_pre_entries", ""),
        "positives_after_filters": positives_stats.get("positives_after_filters", ""),
        "positives_written": positives_stats.get("positives_written", ""),
        "positive_chr_missing": positives_stats.get("skipped_chr_missing", ""),
        "positive_boundary_filtered": positives_stats.get("skipped_boundary", ""),
        "positive_repeat_filtered": positives_stats.get("skipped_repeat", ""),
        "negative_windows_folded": pool_stats.get("folded_candidate_windows", ""),
        "negative_hairpin_like_kept": pool_stats.get("hairpin_like_negatives_kept", ""),
        "negative_overlap_skipped": pool_stats.get("skipped_overlap", ""),
        "negative_repeat_or_n_skipped": pool_stats.get("skipped_repeat_or_n", ""),
        "positives": sum(1 for row in species_rows if row["label"] == "1"),
        "negatives": sum(1 for row in species_rows if row["label"] == "0"),
        "train_pos": counts[("train", "1")],
        "train_neg": counts[("train", "0")],
        "valid_pos": counts[("valid", "1")],
        "valid_neg": counts[("valid", "0")],
        "test_chrom_pos": counts[("test_chrom", "1")],
        "test_chrom_neg": counts[("test_chrom", "0")],
        "test_species_pos": counts[("test_species", "1")],
        "test_species_neg": counts[("test_species", "0")],
        "issues": "; ".join(species_issues),
    }
    return {
        "species_index": species_index,
        "species": species,
        "prefixed_genome": str(prefixed_genome),
        "rows": species_rows,
        "summary": summary_row,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build a prefixed multi-species 1:N training dataset.")
    parser.add_argument("--panel", required=True, help="panel.tsv from benchmark/download/download_diverse20.sh")
    parser.add_argument("--output-dir", default="data/train/diverse20")
    parser.add_argument("--work-dir", default="benchmark/train_data/work_diverse20")
    parser.add_argument("--species", default=None, help="Comma-separated species codes to include. Default: all auto species in panel.")
    parser.add_argument("--heldout-species", default="dre,dme")
    parser.add_argument("--ratio", type=float, default=5.0)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--max-negative-windows-per-species", type=int, default=50000)
    parser.add_argument(
        "--sequential-negative-scan",
        action="store_true",
        help="Scan negative windows in FASTA order. Default balances capped scans across BED-positive chromosomes.",
    )
    parser.add_argument("--max-repeat-frac", type=float, default=0.1)
    parser.add_argument("--min-mfe", type=float, default=-10.0)
    parser.add_argument("--min-paired-frac", type=float, default=0.40)
    parser.add_argument("--min-stem", type=int, default=8)
    parser.add_argument("--max-loop", type=int, default=25)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--species-jobs", type=int, default=1, help="Number of species to process in parallel.")
    parser.add_argument("--mining-rounds", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=10)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--consensus", type=float, default=0.5)
    parser.add_argument(
        "--mining-jobs",
        type=int,
        default=None,
        help="RandomForest jobs per species. Default: -1 sequentially, 1 when --species-jobs > 1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ratio <= 0:
        raise SystemExit("--ratio must be positive")

    script_dir = Path(__file__).resolve().parent
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    species_filter = [item.strip() for item in args.species.split(",") if item.strip()] if args.species else None
    panel_rows = load_panel(args.panel, species_filter)

    combined_genome = output_dir / "genome.fa"
    if combined_genome.exists():
        combined_genome.unlink()

    all_rows = []
    summary_rows = []
    python_exe = sys.executable
    species_jobs = max(1, args.species_jobs)
    mining_jobs = args.mining_jobs
    if mining_jobs is None:
        mining_jobs = 1 if species_jobs > 1 else -1
    print(f"species jobs: {species_jobs}")
    print(f"RNAfold jobs per species: {args.cpus}")
    print(f"RandomForest jobs per species: {mining_jobs}")

    results = []
    if species_jobs == 1:
        for species_index, species_row in enumerate(panel_rows, start=1):
            results.append(
                process_species(
                    species_index,
                    len(panel_rows),
                    species_row,
                    args,
                    script_dir,
                    python_exe,
                    mining_jobs,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=species_jobs) as executor:
            futures = {
                executor.submit(
                    process_species,
                    species_index,
                    len(panel_rows),
                    species_row,
                    args,
                    script_dir,
                    python_exe,
                    mining_jobs,
                ): species_row["code"]
                for species_index, species_row in enumerate(panel_rows, start=1)
            }
            for future in as_completed(futures):
                species = futures[future]
                result = future.result()
                print(f"### completed {species}")
                results.append(result)

    for result in sorted(results, key=lambda row: row["species_index"]):
        append_file(result["prefixed_genome"], combined_genome)
        all_rows.extend(result["rows"])
        summary_rows.append(result["summary"])

    dataset_csv = output_dir / "dataset.csv"
    split_summary = output_dir / "split_summary.csv"
    write_csv(dataset_csv, all_rows)
    write_split_summary(split_summary, summary_rows)
    prepare_tool_inputs(all_rows, output_dir / "tool_inputs", target_lengths=DEFAULT_TARGET_LENGTHS)

    print("\nsummary")
    print(f"species: {len(panel_rows)}")
    print(f"records: {len(all_rows)}")
    print(f"dataset: {dataset_csv}")
    print(f"split summary: {split_summary}")
    print(f"combined genome: {combined_genome}")
    print(f"tool inputs: {output_dir / 'tool_inputs'}")
    issue_rows = [row for row in summary_rows if row.get("issues")]
    if issue_rows:
        print("\nissues")
        for row in issue_rows:
            print(f"{row['species']}: {row['issues']}")
    else:
        print("issues: none")


if __name__ == "__main__":
    main()
