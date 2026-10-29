#!/usr/bin/env python3
"""
How many MirGeneDB loci a method finds among its top-ranked candidates on one held-out genome, for all loci and
separately for loci of training families and of held-out families (figure option, user request 22 Oct).

Candidates, labels and training families as in genome_scale_metrics.py; ambiguous candidates are left out. Candidates
are ranked by score; tied scores form one group, inside which positives are spread evenly (the expected count under a
random order). One row per tie group that contains a positive, plus the last group: candidates ranked before the
group (start), after it (end), and the positives found by its end in each family group.
"""
import argparse
import csv

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--training-families", required=True, help="benchmark dataset.csv")
    parser.add_argument("--method", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.training_families, newline="") as handle:
        training = {row["family_id"] for row in csv.DictReader(handle) if row["split"] == "train" and row["label"] == "1"}
    with open(args.candidates, newline="") as handle:
        candidates = {row["record_id"]: (row["genome_label"], row["family_id"]) for row in csv.DictReader(handle)}
    with open(args.predictions, newline="") as handle:
        scores = {row["record_id"]: float(row["score"]) for row in csv.DictReader(handle)}
    if scores.keys() != candidates.keys():
        raise SystemExit(f"{args.predictions}: record ids differ from {args.candidates}")

    kept = [(scores[r], label, family) for r, (label, family) in candidates.items() if label != "ambiguous"]
    values = np.array([score for score, _, _ in kept])
    is_training = np.array([label == "positive" and family in training for _, label, family in kept])
    is_heldout = np.array([label == "positive" and family not in training for _, label, family in kept])
    order = np.argsort(-values, kind="stable")
    values, is_training, is_heldout = values[order], is_training[order], is_heldout[order]
    boundaries = np.flatnonzero(np.r_[values[1:] != values[:-1], True]) + 1  # end of each tie group
    starts = np.r_[0, boundaries[:-1]]
    found_training = np.cumsum(is_training)[boundaries - 1]
    found_heldout = np.cumsum(is_heldout)[boundaries - 1]
    with_positive = np.r_[found_training[0] + found_heldout[0] > 0,
                          np.diff(found_training + found_heldout) > 0]
    with_positive[-1] = True
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "genome", "start", "end", "found_training", "found_heldout",
                         "total_training", "total_heldout", "candidates"])
        for i in np.flatnonzero(with_positive):
            writer.writerow([args.method, args.genome, starts[i], boundaries[i], found_training[i], found_heldout[i],
                             int(is_training.sum()), int(is_heldout.sum()), len(values)])
    print(f"{args.method} {args.genome}: {int(with_positive.sum())} rows, {int(is_training.sum())} training-family and "
          f"{int(is_heldout.sum())} held-out-family positives")


if __name__ == "__main__":
    main()
