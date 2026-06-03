#!/usr/bin/env python3
"""
Build a structurally plausible negative pool from genome windows.

The script scans fixed-size windows, excludes windows that overlap known
pre-miRNA loci, folds the remaining sequences with RNAfold, and keeps only
hairpin-like folds. Output rows share the dataset CSV schema used by the rest
of benchmark/train_data.
"""
import argparse
import csv
import math
import os
import subprocess
import tempfile
from collections import defaultdict


COMPLEMENT = str.maketrans("ACGUTRYSWKMBDHVNacgutryswkmbdhvn", "UGCAAYRSWMKVHDBNUGCAAYRSWMKVHDBN")


def reverse_complement(seq):
    return seq.translate(COMPLEMENT)[::-1].upper()


def count_masked(seq):
    return sum(1 for char in seq if char == "N" or char.islower())


def iter_fasta(path):
    header = None
    chunks = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def load_excluded_intervals(path, suffix):
    intervals = defaultdict(list)
    with open(path) as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.startswith("#"):
                continue
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[3] if len(parts) > 3 else ""
            if suffix and not name.endswith(suffix):
                continue
            intervals[parts[0]].append((int(parts[1]), int(parts[2])))

    for chrom in intervals:
        intervals[chrom].sort()
    return intervals


def overlaps_any(start, end, intervals):
    for interval_start, interval_end in intervals:
        if start < interval_end and interval_start < end:
            return True
        if interval_start >= end:
            break
    return False


def parse_rnafold_output(text):
    rows = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index in range(0, len(lines), 3):
        if index + 2 >= len(lines) or not lines[index].startswith(">"):
            continue
        window_id = lines[index][1:].strip()
        folded_sequence = lines[index + 1]
        struct_line = lines[index + 2]
        left = struct_line.rfind("(")
        right = struct_line.rfind(")")
        if left == -1 or right == -1 or right <= left:
            continue
        try:
            mfe = float(struct_line[left + 1:right].strip())
        except ValueError:
            continue
        rows[window_id] = (folded_sequence, struct_line[:left].strip(), mfe)
    return rows


def fold_batch(batch, cpus):
    if not batch:
        return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "windows.fa")
        with open(fasta_path, "w") as handle:
            for row in batch:
                handle.write(f">{row['window_id']}\n{row['sequence']}\n")

        cmd = ["RNAfold", "--noPS", f"--jobs={cpus}"]
        with open(fasta_path) as input_handle:
            process = subprocess.run(
                cmd,
                stdin=input_handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
    return parse_rnafold_output(process.stdout)


def longest_run(text, chars):
    best = 0
    current = 0
    for char in text:
        if char in chars:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def hairpin_features(structure):
    if not structure:
        return 0.0, 0, 0
    paired = structure.count("(") + structure.count(")")
    paired_frac = paired / len(structure)
    max_stem = max(longest_run(structure, "("), longest_run(structure, ")"))
    max_loop = longest_run(structure, ".")
    return paired_frac, max_stem, max_loop


def passes_filters(row, min_mfe, min_paired_frac, min_stem, max_loop):
    paired_frac, max_stem, loop_size = hairpin_features(row["structure"])
    return (
        float(row["mfe"]) <= min_mfe
        and paired_frac >= min_paired_frac
        and max_stem >= min_stem
        and loop_size <= max_loop
    )


def write_rows(output_path, rows, write_header):
    fieldnames = [
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
    ]
    mode = "a" if os.path.exists(output_path) else "w"
    with open(output_path, mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract hairpin-like negative windows from a genome FASTA.")
    parser.add_argument("--genome", required=True)
    parser.add_argument("--bed", required=True, help="Known miRNA BED used as an exclusion mask.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--chr", dest="chromosomes", default=None, help="Comma-separated chromosomes to scan.")
    parser.add_argument("--bed-name-suffix", default="_pre", help="Only exclude BED names ending with this suffix. Empty = all rows.")
    parser.add_argument("--max-repeat-frac", type=float, default=0.1)
    parser.add_argument("--min-mfe", type=float, default=-10.0)
    parser.add_argument("--min-paired-frac", type=float, default=0.40)
    parser.add_argument("--min-stem", type=int, default=8)
    parser.add_argument("--max-loop", type=int, default=25)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--max-windows", type=int, default=0, help="Stop after folding this many candidate windows. 0 = no cap.")
    parser.add_argument("--single-strand", action="store_true", help="Scan only the forward strand.")
    parser.add_argument(
        "--balance-bed-chroms",
        action="store_true",
        help="When max-windows is capped, divide the scan across BED-positive chromosomes instead of scanning FASTA order.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)

    chromosome_filter = None
    if args.chromosomes:
        chromosome_filter = {chrom.strip() for chrom in args.chromosomes.split(",") if chrom.strip()}

    excluded = load_excluded_intervals(args.bed, args.bed_name_suffix)
    per_chrom_max_windows = 0
    if args.balance_bed_chroms:
        bed_chroms = set(excluded)
        if chromosome_filter:
            bed_chroms &= chromosome_filter
        if bed_chroms:
            chromosome_filter = bed_chroms
            if args.max_windows:
                per_chrom_max_windows = max(1, math.ceil(args.max_windows / len(bed_chroms)))

    batch = []
    folded = 0
    kept = 0
    skipped_repeat = 0
    skipped_overlap = 0
    skipped_chrom = 0
    wrote_header = False

    def flush_batch():
        nonlocal batch, folded, kept, wrote_header
        fold_results = fold_batch(batch, args.cpus)
        folded += len(batch)
        output_rows = []
        for row in batch:
            result = fold_results.get(row["window_id"])
            if not result:
                continue
            folded_sequence, structure, mfe = result
            row["sequence"] = folded_sequence.upper().replace("T", "U")
            row["structure"] = structure
            row["mfe"] = f"{mfe:.6g}"
            if passes_filters(row, args.min_mfe, args.min_paired_frac, args.min_stem, args.max_loop):
                output_rows.append(row)
        if output_rows:
            write_rows(args.output, output_rows, not wrote_header)
            wrote_header = True
            kept += len(output_rows)
        batch = []

    for chrom, chrom_sequence in iter_fasta(args.genome):
        if chromosome_filter and chrom not in chromosome_filter:
            skipped_chrom += 1
            continue
        chrom_intervals = excluded.get(chrom, [])
        chrom_windows = 0
        max_start = len(chrom_sequence) - args.window
        if max_start < 0:
            continue

        for start in range(0, max_start + 1, args.step):
            end = start + args.window
            if overlaps_any(start, end, chrom_intervals):
                skipped_overlap += 1
                continue

            raw_window = chrom_sequence[start:end]
            if count_masked(raw_window) / args.window > args.max_repeat_frac:
                skipped_repeat += 1
                continue

            strands = [("+", raw_window)]
            if not args.single_strand:
                strands.append(("-", reverse_complement(raw_window)))

            for strand, strand_sequence in strands:
                sequence = strand_sequence.upper().replace("T", "U")
                if "N" in sequence:
                    skipped_repeat += 1
                    continue
                batch.append(
                    {
                        "window_id": f"{chrom}|{start + 1}-{end}|{strand}",
                        "chrom": chrom,
                        "start": str(start),
                        "end": str(end),
                        "strand": strand,
                        "sequence": sequence,
                        "structure": "",
                        "mfe": "",
                        "mirna_id": "",
                        "target_start": "",
                        "target_end": "",
                        "label": "0",
                    }
                )
                chrom_windows += 1

                if len(batch) >= args.batch_size:
                    flush_batch()
                if per_chrom_max_windows and chrom_windows >= per_chrom_max_windows:
                    break
                if not per_chrom_max_windows and args.max_windows and folded + len(batch) >= args.max_windows:
                    break
            if per_chrom_max_windows and chrom_windows >= per_chrom_max_windows:
                break
            if not per_chrom_max_windows and args.max_windows and folded + len(batch) >= args.max_windows:
                break
        if per_chrom_max_windows and batch:
            flush_batch()
        if not per_chrom_max_windows and args.max_windows and folded + len(batch) >= args.max_windows:
            break

    if batch:
        flush_batch()

    if not wrote_header:
        write_rows(args.output, [], True)

    print(f"folded candidate windows: {folded}")
    print(f"hairpin-like negatives kept: {kept}")
    print(f"skipped overlap: {skipped_overlap}")
    print(f"skipped repeat/N: {skipped_repeat}")
    if chromosome_filter:
        print(f"chromosomes skipped: {skipped_chrom}")
    if per_chrom_max_windows:
        print(f"balanced BED chromosomes: {len(chromosome_filter)}")
        print(f"max windows per BED chromosome: {per_chrom_max_windows}")
    print(f"output: {args.output}")

    if args.stats_output:
        with open(args.stats_output, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            writer.writerow(["folded_candidate_windows", folded])
            writer.writerow(["hairpin_like_negatives_kept", kept])
            writer.writerow(["skipped_overlap", skipped_overlap])
            writer.writerow(["skipped_repeat_or_n", skipped_repeat])
            writer.writerow(["chromosomes_skipped", skipped_chrom])
            writer.writerow(["balanced_bed_chromosomes", len(chromosome_filter) if per_chrom_max_windows else 0])
            writer.writerow(["max_windows_per_bed_chromosome", per_chrom_max_windows])


if __name__ == "__main__":
    main()
