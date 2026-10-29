#!/usr/bin/env python3
"""
Genome-scale AP of one method on one held-out genome, separately for MirGeneDB loci of training families and of
held-out families.

Candidates and labels as in genome_scale_metrics.py (ambiguous candidates left out; a positive is a candidate matched to
a MirGeneDB precursor, its family that precursor's family). Training families: families of training positives in the
benchmark dataset.csv, as in genome_scale_metrics.py. Training-family AP: all negatives and the positives of training
families (held-out-family positives left out); held-out-family AP: all negatives and the positives of held-out families.
The AP over all positives is repeated as a check against the genome metrics tables.
"""
import argparse
import csv

import numpy as np
from sklearn.metrics import average_precision_score


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

    group, values = [], []
    for record_id, (label, family) in candidates.items():
        if label == "ambiguous":
            continue
        if label == "positive" and not family:
            raise SystemExit(f"{record_id}: positive without a family")
        group.append("negative" if label != "positive" else "training" if family in training else "heldout")
        values.append(scores[record_id])
    group, values = np.array(group), np.array(values)
    row = {"method": args.method, "genome": args.genome, "negatives": int((group == "negative").sum()),
           "positives": int((group != "negative").sum()), "positives_training_family": int((group == "training").sum()),
           "positives_heldout_family": int((group == "heldout").sum()),
           "ap_unambiguous_candidates": average_precision_score(group != "negative", values)}
    for name in ("training", "heldout"):
        keep = group != ("heldout" if name == "training" else "training")
        row[f"ap_{name}_family_loci"] = average_precision_score(group[keep] == name, values[keep])
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"{args.method} {args.genome}: AP all {row['ap_unambiguous_candidates']:.4f}, training families "
          f"{row['ap_training_family_loci']:.4f} ({row['positives_training_family']}), held-out families "
          f"{row['ap_heldout_family_loci']:.4f} ({row['positives_heldout_family']})")


if __name__ == "__main__":
    main()
