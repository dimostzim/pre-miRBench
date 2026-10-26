#!/usr/bin/env python3
"""
Genome-scale metrics for one held-out genome.

Inputs: candidates.csv and loci.csv from prepare_genome_candidates.py, record-level
genome predictions (tool, record_id, score) and validation predictions (tool,
label, score) for the same predictors.

Thresholds are fixed on the validation partition, never on the genome:
  valid_max_f1      threshold maximizing validation F1
  valid_recall_90   highest threshold reaching 90% validation recall

For each predictor and threshold the script reports calls (all candidates at or
above threshold, ambiguous ones included) per Mb, locus recall over all annotated
loci and over loci with a matched candidate, and precision among unambiguous
calls. AP is computed over positive and negative candidates (ambiguous ones
excluded) at the true genomic prevalence. Top-N recall is the fraction of annotated loci
whose matched candidate ranks among the N highest-scoring unambiguous candidates, for
N = 1x, 2x and 10x the number of annotated loci; candidates tied at the cutoff score count
fractionally (the expected value under a random order of tied candidates). Locus recall is
reported over all loci and over unmasked loci (the masked flag of loci.csv from
prepare_genome_candidates.py: the benchmark's repeat rule applied to the precursor),
the latter also split by whether the family is represented in benchmark training.
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def validation_thresholds(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    reaching = np.flatnonzero(recall[:-1] >= 0.90)
    return {
        "valid_max_f1": float(thresholds[int(np.argmax(f1))]),
        "valid_recall_90": float(thresholds[reaching.max()]) if reaching.size else float(thresholds.min()),
    }


def top_n_inclusion(values, n):
    """Probability that each candidate ranks among the n highest values when ties are ordered at random."""
    if n >= len(values):
        return np.ones(len(values))
    cutoff = np.partition(values, len(values) - n)[len(values) - n]
    above = values > cutoff
    tied = values == cutoff
    return np.where(above, 1.0, np.where(tied, (n - above.sum()) / tied.sum(), 0.0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--loci", required=True)
    parser.add_argument("--genome-predictions", required=True, nargs="+")
    parser.add_argument("--validation-predictions", required=True, nargs="+")
    parser.add_argument("--training-families", required=True, help="Benchmark dataset.csv; families of training positives.")
    parser.add_argument("--genome-bases", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = {row["record_id"]: row for row in read_csv(args.candidates)}
    loci = read_csv(args.loci)
    training_families = {
        row["family_id"] for row in read_csv(args.training_families) if row["split"] == "train" and row["label"] == "1"
    }
    genome_scores = defaultdict(dict)
    for path in args.genome_predictions:
        for row in read_csv(path):
            genome_scores[row["tool"]][row["record_id"]] = float(row["score"])
    validation = defaultdict(lambda: ([], []))
    for path in args.validation_predictions:
        for row in read_csv(path):
            if row.get("split", "valid") != "valid":
                continue  # evaluation files also hold test rows; thresholds come from validation only
            labels, scores = validation[row["tool"]]
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))

    megabases = args.genome_bases / 1e6
    unmasked = [locus for locus in loci if locus["masked"] == "0"]
    locus_groups = {
        "all_loci": loci,
        "unmasked_loci": unmasked,
        "unmasked_training_family_loci": [locus for locus in unmasked if locus["family_id"] in training_families],
        "unmasked_heldout_family_loci": [locus for locus in unmasked if locus["family_id"] not in training_families],
    }
    rows = []
    for tool, scores in sorted(genome_scores.items()):
        missing = set(candidates) - set(scores)
        if missing:
            raise ValueError(f"{tool}: {len(missing)} candidates without a genome prediction")
        ids = list(candidates)
        values = np.array([scores[record_id] for record_id in ids])
        status = np.array([candidates[record_id]["genome_label"] for record_id in ids])
        scored = status != "ambiguous"
        y = (status[scored] == "positive").astype(int)
        ap = average_precision_score(y, values[scored])
        scored_index = {record_id: i for i, record_id in enumerate(np.array(ids)[scored])}
        matched = [scored_index[locus["matched_record_id"]] for locus in loci if locus["matched_record_id"] in scored_index]
        top = {f"recall_top_{k}x_loci": float(top_n_inclusion(values[scored], k * len(loci))[matched].sum() / len(loci)) for k in (1, 2, 10)}
        thresholds = validation_thresholds(np.array(validation[tool][0]), np.array(validation[tool][1]))
        for name, threshold in thresholds.items():
            called = values >= threshold
            unambiguous_calls = called & scored
            true_calls = called & (status == "positive")
            called_ids = {record_id for record_id, flag in zip(ids, true_calls) if flag}
            row = {
                "tool": tool,
                "threshold_rule": name,
                "threshold": threshold,
                "candidates": len(ids),
                "ap_unambiguous_candidates": ap,
                "calls": int(called.sum()),
                "calls_per_mb": float(called.sum() / megabases),
                "precision_unambiguous_calls": float(true_calls.sum() / unambiguous_calls.sum()) if unambiguous_calls.any() else float("nan"),
                **top,
            }
            for group, members in locus_groups.items():
                matched = [locus for locus in members if locus["matched_record_id"]]
                recalled = sum(1 for locus in members if locus["matched_record_id"] in called_ids)
                row[f"{group}"] = len(members)
                row[f"{group}_with_candidate"] = len(matched)
                row[f"{group}_recall"] = recalled / len(members) if members else float("nan")
                row[f"{group}_recall_of_candidates"] = recalled / len(matched) if matched else float("nan")
            rows.append(row)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
