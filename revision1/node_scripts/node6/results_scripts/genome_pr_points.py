#!/usr/bin/env python3
"""
Genome-scale precision-recall curve of one method on one held-out genome, for the genome precision-recall figure.

Candidates and labels as in genome_scale_metrics.py: ambiguous candidates are left out, and a candidate is positive
when prepare_genome_candidates.py matched it to a MirGeneDB precursor. Scores come from a genome_eval predictions.csv
(predictors) or a compare_annotation_calls.py --predictions file (MirMachine, ncOrtho; candidates without a hit share
the lowest score). Of scikit-learn's curve only the points where recall rises are kept, plus the end point at recall
0: they give the AP exactly and the same step plot, with at most one point per positive.
"""
import argparse
import csv

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.candidates, newline="") as handle:
        status = {row["record_id"]: row["genome_label"] for row in csv.DictReader(handle)}
    with open(args.predictions, newline="") as handle:
        scores = {row["record_id"]: float(row["score"]) for row in csv.DictReader(handle)}
    if scores.keys() != status.keys():
        raise SystemExit(f"{args.predictions}: record ids differ from {args.candidates}")
    ids = [record_id for record_id, label in status.items() if label != "ambiguous"]
    labels = np.array([status[record_id] == "positive" for record_id in ids], dtype=int)
    values = np.array([scores[record_id] for record_id in ids])

    precision, recall, thresholds = precision_recall_curve(labels, values)
    keep = np.flatnonzero(recall[:-1] > recall[1:])
    ap = average_precision_score(labels, values)
    kept_ap = float(np.sum((recall[keep] - recall[keep + 1]) * precision[keep]))
    if abs(kept_ap - ap) > 1e-9:
        raise SystemExit(f"kept points give AP {kept_ap}, average_precision_score {ap}")
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "genome", "threshold", "precision", "recall", "positives", "negatives", "ap"])
        for i in keep:
            writer.writerow([args.method, args.genome, thresholds[i], precision[i], recall[i], labels.sum(), len(labels) - labels.sum(), ap])
        writer.writerow([args.method, args.genome, "", 1.0, 0.0, labels.sum(), len(labels) - labels.sum(), ap])
    print(f"{args.method} {args.genome}: AP {ap:.6f}, {len(keep) + 1} points, {labels.sum()} positives, {len(labels) - labels.sum()} negatives")


if __name__ == "__main__":
    main()
