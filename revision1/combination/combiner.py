#!/usr/bin/env python3
"""Combiner of pre-miRBench and retrained MirMachine (29 Oct 2026; choices fixed in STATUS.md 6a).

Logistic regression (scikit-learn defaults on standardized inputs) on four inputs per record: the pre-miRBench
log-odds, a MirMachine hit flag, MirMachine's score (bitscore minus the family cutoff; 0 without a hit) and a
MirMachine call flag. The combined score is the fitted log-odds.

table      joins the pre-miRBench and MirMachine predictions of one split by record id
fit        fits the combiner on one or more tables (validation data only) and writes it as JSON
apply      writes the combined score of one split in the predictors' format (tool, split, record_id, species, label, score)
threshold  maximum F1 on weighted tables (column weight) for the combiner and for pre-miRBench alone, written as a
           thresholds CSV (tool, threshold_rule, threshold) for genome_scale_metrics.py --thresholds
"""
import argparse
import csv
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

NO_HIT_SCORE = -1000.0
INPUTS = ["premirbench_log_odds", "mirmachine_hit", "mirmachine_score", "mirmachine_called"]


def read_split(path, split):
    with open(path, newline="") as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle) if row["split"] == split}


def joined(premirbench_path, mirmachine_path, split):
    premirbench, mirmachine = read_split(premirbench_path, split), read_split(mirmachine_path, split)
    if not premirbench or premirbench.keys() != mirmachine.keys():
        raise SystemExit(f"{split}: record ids differ between {premirbench_path} and {mirmachine_path}")
    for record_id, row in premirbench.items():
        other = mirmachine[record_id]
        if other["label"] != row["label"]:
            raise SystemExit(f"{record_id}: labels differ")
        yield {"record_id": record_id, "species": row["species"], "label": row["label"], "premirbench_score": row["score"],
               "mirmachine_score": other["score"], "mirmachine_called": other["called"]}


def features(rows):
    premirbench = np.array([float(r["premirbench_score"]) for r in rows])
    if ((premirbench < 0) | (premirbench > 1)).any():
        raise SystemExit("pre-miRBench scores outside [0, 1]")
    p = np.clip(premirbench, 1e-7, 1 - 1e-7)
    margin = np.array([float(r["mirmachine_score"]) for r in rows])
    hit = margin > NO_HIT_SCORE
    called = np.array([int(r["mirmachine_called"]) for r in rows])
    return np.column_stack([np.log(p / (1 - p)), hit, np.where(hit, margin, 0.0), called]).astype(float)


def combined_score(model, x):
    z = (x - np.array(model["mean"])) / np.array(model["scale"])
    return z @ np.array(model["coef"]) + model["intercept"]


def read_tables(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def table(args):
    rows = list(joined(args.premirbench, args.mirmachine, args.split))
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{args.split}: {len(rows)} records -> {args.output}")


def fit(args):
    rows = read_tables(args.tables)
    x, y = features(rows), np.array([int(r["label"]) for r in rows])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    model = LogisticRegression(max_iter=5000).fit((x - mean) / scale, y)
    fitted = {"inputs": INPUTS, "mean": mean.tolist(), "scale": scale.tolist(), "coef": model.coef_[0].tolist(),
              "intercept": float(model.intercept_[0]), "records": len(y), "positives": int(y.sum()), "tables": args.tables}
    fitted["ap_on_fitting_data"] = float(average_precision_score(y, combined_score(fitted, x)))
    with open(args.output, "w") as handle:
        json.dump(fitted, handle, indent=1)
    print(json.dumps({k: fitted[k] for k in ("inputs", "coef", "intercept", "records", "positives", "ap_on_fitting_data")}))


def apply(args):
    with open(args.model) as handle:
        model = json.load(handle)
    rows = list(joined(args.premirbench, args.mirmachine, args.split))
    scores = combined_score(model, features(rows))
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tool", "split", "record_id", "species", "label", "score"])
        for row, score in zip(rows, scores):
            writer.writerow([args.tool, args.split, row["record_id"], row["species"], row["label"], repr(float(score))])
    print(f"{args.split}: {len(rows)} records -> {args.output}")


def max_f1_threshold(labels, scores, weights):
    """As genome_scale_metrics.validation_thresholds (valid_max_f1), with sample weights."""
    precision, recall, thresholds = precision_recall_curve(labels, scores, sample_weight=weights)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(precision[best]), float(recall[best])


def threshold(args):
    with open(args.model) as handle:
        model = json.load(handle)
    rows = read_tables(args.tables)
    labels, weights = np.array([int(r["label"]) for r in rows]), np.array([float(r["weight"]) for r in rows])
    candidates = {args.combiner_tool: combined_score(model, features(rows)),
                  args.premirbench_tool: np.array([float(r["premirbench_score"]) for r in rows])}
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tool", "threshold_rule", "threshold", "weighted_precision", "weighted_recall"])
        for tool, scores in candidates.items():
            value, precision, recall = max_f1_threshold(labels, scores, weights)
            writer.writerow([tool, "genome_validation_max_f1", repr(value), precision, recall])
            print(f"{tool}: threshold {value:.6g}, weighted precision {precision:.3f}, recall {recall:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("table", "apply"):
        command = sub.add_parser(name)
        command.add_argument("--premirbench", required=True)
        command.add_argument("--mirmachine", required=True)
        command.add_argument("--split", required=True)
        command.add_argument("--output", required=True)
    sub.choices["apply"].add_argument("--model", required=True)
    sub.choices["apply"].add_argument("--tool", required=True)
    command = sub.add_parser("fit")
    command.add_argument("--tables", nargs="+", required=True)
    command.add_argument("--output", required=True)
    command = sub.add_parser("threshold")
    command.add_argument("--model", required=True)
    command.add_argument("--tables", nargs="+", required=True)
    command.add_argument("--combiner-tool", required=True)
    command.add_argument("--premirbench-tool", required=True)
    command.add_argument("--output", required=True)
    args = parser.parse_args()
    {"table": table, "fit": fit, "apply": apply, "threshold": threshold}[args.command](args)


if __name__ == "__main__":
    main()
