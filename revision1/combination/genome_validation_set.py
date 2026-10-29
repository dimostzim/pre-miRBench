#!/usr/bin/env python3
"""Genome-scale validation table of one training species for the combiner (29 Oct 2026; STATUS.md 6a, item 4).

Candidates and labels from prepare_genome_candidates.py. Label 1: a positive candidate whose hairpin overlaps (same
strand) only validation precursors of the benchmark. Left out: ambiguous candidates, positive candidates overlapping any
other precursor (training or test precursors, or precursors outside the benchmark), and every candidate whose hairpin
overlaps a training or test window of the benchmark (either strand). Weight: 1 for label 0; for label 1, the species'
positive candidates of all precursors divided by its label-1 candidates, so the weighted positives stand for all
precursors of the genome. Writes the combiner.py table columns plus weight.
"""
import argparse
import csv
from bisect import bisect_left
from collections import Counter, defaultdict


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def by_record(path):
    return {row["record_id"]: row for row in read_csv(path)}


def index(intervals):
    """(chrom -> (sorted starts, intervals sorted by start), length of the longest interval)."""
    grouped, longest = defaultdict(list), 0
    for chrom, start, end, payload in intervals:
        grouped[chrom].append((start, end, payload))
        longest = max(longest, end - start)
    items = {chrom: sorted(v, key=lambda item: item[0]) for chrom, v in grouped.items()}
    return {chrom: ([s for s, _, _ in v], v) for chrom, v in items.items()}, longest


def overlapping(indexed, chrom, start, end):
    idx, longest = indexed
    if chrom not in idx:
        return []
    starts, items = idx[chrom]
    i = bisect_left(starts, start - longest)
    found = []
    while i < len(items) and items[i][0] < end:
        if items[i][1] > start:
            found.append(items[i][2])
        i += 1
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--species", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--loci", required=True)
    parser.add_argument("--dataset", required=True, help="benchmark dataset.csv")
    parser.add_argument("--premirbench", required=True)
    parser.add_argument("--mirmachine", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = [row for row in read_csv(args.dataset) if row["species"] == args.species]
    validation = {row["mirna_id"] for row in dataset if row["split"] == "valid" and row["label"] == "1"}
    windows = index((row["chrom"], int(row["start"]), int(row["end"]), row["record_id"]) for row in dataset if row["split"] != "valid")
    loci = read_csv(args.loci)
    precursors = index((l["chrom"], int(l["start"]), int(l["end"]), (l["strand"], l["name"])) for l in loci)
    premirbench, mirmachine = by_record(args.premirbench), by_record(args.mirmachine)

    counts, rows = Counter(), []
    for candidate in read_csv(args.candidates):
        record_id, status = candidate["record_id"], candidate["genome_label"]
        chrom, start, end = candidate["chrom"], int(candidate["hairpin_start"]), int(candidate["hairpin_end"])
        counts[f"candidates_{status}"] += 1
        if status == "ambiguous":
            continue
        if overlapping(windows, chrom, start, end):
            counts[f"left_out_{status}_on_training_or_test_window"] += 1
            continue
        if status == "positive":
            names = {name for strand, name in overlapping(precursors, chrom, start, end) if strand == candidate["strand"]}
            if not names or not names <= validation:
                counts["left_out_positive_not_only_validation_precursors"] += 1
                continue
        p, m = premirbench[record_id], mirmachine[record_id]
        rows.append({"record_id": record_id, "species": args.species, "label": int(status == "positive"),
                     "premirbench_score": p["score"], "mirmachine_score": m["score"], "mirmachine_called": m["called"]})
    if len(premirbench) != sum(v for k, v in counts.items() if k.startswith("candidates_")) or premirbench.keys() != mirmachine.keys():
        raise SystemExit(f"{args.species}: prediction files do not cover the candidates")

    positives = sum(r["label"] for r in rows)
    weight = counts["candidates_positive"] / positives if positives else 1.0
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "species", "label", "weight", "premirbench_score", "mirmachine_score",
                                                    "mirmachine_called"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "weight": weight if row["label"] else 1.0})
    print(f"{args.species}: {dict(counts)}; kept {len(rows)}, label 1 {positives} (weight {weight:.2f}) of "
          f"{len(validation)} validation precursors -> {args.output}")


if __name__ == "__main__":
    main()
