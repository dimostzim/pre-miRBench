#!/usr/bin/env python3
"""
Summarize benchmark evaluations over training replicates.

Inputs
  --predictions  predictions.csv files (columns tool, split, record_id, label, score), one per
                 trained model; evaluate_replicates.sh writes <out>/<tool>_<run>/predictions.csv.
                 The replicate name is the file's parent directory. Each file must hold the
                 valid split, which fixes the thresholds, and the test splits.

Thresholds are fixed per replicate on its validation predictions (valid_max_f1,
valid_recall_90; same definitions as genome_scale_metrics.py); fixed_0.5 is kept as a
secondary rule and is meaningful only for probability scores.

Outputs (--output-dir)
  replicate_metrics.csv  per tool, replicate, split and group (held-out species for the
                         held-out-species tests; "" = whole split): AP, AUROC, and threshold,
                         precision, recall, F1 and MCC for each threshold rule
  summary.csv            mean and SD over replicates of every metric; for whole splits, the
                         95% percentile bootstrap CI of the replicate-mean AP
  paired_differences.csv for every pair of tools and test split: difference in replicate-mean
                         AP (tool_a minus tool_b) with its 95% percentile bootstrap CI

Bootstrap: class-stratified resamples of each test split (positives and negatives resampled
separately with replacement), the same record indices for every model. Intervals are
conditional on the fitted models and the fixed test composition.
"""
import argparse
import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "genome_scale"))
from genome_scale_metrics import validation_thresholds  # noqa: E402

TEST_SPLITS = [
    "test_known_species_known_family",
    "test_known_species_heldout_family",
    "test_heldout_species_known_family",
    "test_heldout_species_heldout_family",
]
HELDOUT_SPECIES_SPLITS = {"test_heldout_species_known_family", "test_heldout_species_heldout_family"}
THRESHOLD_METRICS = ["threshold", "precision", "recall", "f1", "mcc"]


def average_precision_rows(labels, scores):
    """AP of each row of scores (models x records), with tied scores grouped as in
    sklearn.metrics.average_precision_score: sum over distinct thresholds of recall gain x precision."""
    order = np.argsort(-scores, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    true_positives = np.cumsum(labels[order], axis=1)
    last_of_tie = np.ones_like(sorted_scores, dtype=bool)
    last_of_tie[:, :-1] = sorted_scores[:, :-1] != sorted_scores[:, 1:]
    at_threshold = np.where(last_of_tie, true_positives, 0)
    previous = np.zeros_like(at_threshold)
    previous[:, 1:] = np.maximum.accumulate(at_threshold, axis=1)[:, :-1]
    gain = np.where(last_of_tie, true_positives - previous, 0)
    precision = true_positives / np.arange(1, scores.shape[1] + 1)
    return (gain * precision).sum(axis=1) / labels.sum()


def auroc(labels, scores):
    ranks = np.empty(len(scores))
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    ends = np.r_[starts[1:], len(scores)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2.0
    positives = labels.sum()
    negatives = len(labels) - positives
    return (ranks[labels == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)


def threshold_metrics(labels, scores, threshold):
    called = scores >= threshold
    tp = int((called & (labels == 1)).sum())
    fp = int((called & (labels == 0)).sum())
    fn = int((~called & (labels == 1)).sum())
    tn = int((~called & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else float("nan")
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "mcc": mcc}


def read_predictions(paths):
    """{(tool, replicate): {split: {record_id: (label, score, species)}}}"""
    models = {}
    for path in paths:
        path = Path(path)
        replicate = path.parent.name
        splits = defaultdict(dict)
        tools = set()
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if row["score"] == "":
                    raise ValueError(f"{path}: record {row['record_id']} in {row['split']} has no score")
                tools.add(row["tool"])
                splits[row["split"]][row["record_id"]] = (int(row["label"]), float(row["score"]), row.get("species", ""))
        if len(tools) != 1:
            raise ValueError(f"{path}: expected one tool, found {sorted(tools)}")
        key = (tools.pop(), replicate)
        if key in models:
            raise ValueError(f"duplicate model {key}")
        missing = {"valid", *TEST_SPLITS} - set(splits)
        if missing:
            raise ValueError(f"{path}: missing splits {sorted(missing)}")
        models[key] = splits
    return models


def aligned_split(models, split):
    """Labels, species and a models x records score matrix over the records shared by every model."""
    keys = sorted(models)
    record_ids = sorted(models[keys[0]][split])
    for key in keys[1:]:
        if set(models[key][split]) != set(record_ids):
            raise ValueError(f"{key} scored a different record set for {split} than {keys[0]}")
    reference = models[keys[0]][split]
    labels = np.array([reference[r][0] for r in record_ids])
    for key in keys[1:]:
        if any(models[key][split][r][0] != reference[r][0] for r in record_ids):
            raise ValueError(f"{key} has different labels for {split} than {keys[0]}")
    species = np.array([reference[r][2] for r in record_ids])
    scores = np.array([[models[key][split][r][1] for r in record_ids] for key in keys])
    return keys, labels, species, scores


def mean_sd(values):
    values = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    models = read_predictions(args.predictions)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rules = {}
    for key, splits in models.items():
        valid = splits["valid"].values()
        thresholds = validation_thresholds(np.array([v[0] for v in valid]), np.array([v[1] for v in valid]))
        rules[key] = {**thresholds, "fixed_0.5": 0.5}

    replicate_rows = []
    bootstrap_rows = defaultdict(dict)
    difference_rows = []
    rng = np.random.default_rng(args.seed)
    for split in TEST_SPLITS:
        keys, labels, species, scores = aligned_split(models, split)
        groups = {"": np.ones(len(labels), dtype=bool)}
        if split in HELDOUT_SPECIES_SPLITS:
            groups.update({name: species == name for name in sorted(set(species))})
        for group, mask in groups.items():
            group_labels = labels[mask]
            group_ap = average_precision_rows(group_labels, scores[:, mask])
            for index, (tool, replicate) in enumerate(keys):
                row = {"tool": tool, "replicate": replicate, "split": split, "group": group,
                       "positives": int(group_labels.sum()), "negatives": int((group_labels == 0).sum()),
                       "ap": float(group_ap[index]), "auroc": float(auroc(group_labels, scores[index, mask]))}
                for rule, threshold in rules[(tool, replicate)].items():
                    for name, value in threshold_metrics(group_labels, scores[index, mask], threshold).items():
                        row[f"{rule}_{name}"] = value
                replicate_rows.append(row)

        # Bootstrap the replicate-mean AP of every tool on the same class-stratified resamples.
        tools = sorted({tool for tool, _ in keys})
        rows_of = {tool: [i for i, (t, _) in enumerate(keys) if t == tool] for tool in tools}
        positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
        resampled = np.empty((args.bootstrap, len(tools)))
        for b in range(args.bootstrap):
            index = np.concatenate([rng.choice(positives, len(positives)), rng.choice(negatives, len(negatives))])
            ap = average_precision_rows(labels[index], scores[:, index])
            resampled[b] = [ap[rows_of[tool]].mean() for tool in tools]
        observed = {tool: float(average_precision_rows(labels, scores[rows_of[tool]]).mean()) for tool in tools}
        for column, tool in enumerate(tools):
            low, high = np.percentile(resampled[:, column], [2.5, 97.5])
            bootstrap_rows[(tool, split)] = {"ap_ci_low": float(low), "ap_ci_high": float(high)}
        for (a, tool_a), (b, tool_b) in combinations(enumerate(tools), 2):
            low, high = np.percentile(resampled[:, a] - resampled[:, b], [2.5, 97.5])
            difference_rows.append({"split": split, "tool_a": tool_a, "tool_b": tool_b,
                                    "replicates_a": len(rows_of[tool_a]), "replicates_b": len(rows_of[tool_b]),
                                    "ap_difference": observed[tool_a] - observed[tool_b],
                                    "ci_low": float(low), "ci_high": float(high), "resamples": args.bootstrap, "seed": args.seed})
        print(f"{split}: {len(keys)} models, {len(labels)} records, {args.bootstrap} resamples", flush=True)

    metric_names = [name for name in replicate_rows[0] if name not in {"tool", "replicate", "split", "group", "positives", "negatives"}]
    grouped = defaultdict(list)
    for row in replicate_rows:
        grouped[(row["tool"], row["split"], row["group"])].append(row)
    summary_rows = []
    for (tool, split, group), rows in sorted(grouped.items(), key=lambda item: (item[0][0], TEST_SPLITS.index(item[0][1]), item[0][2])):
        summary = {"tool": tool, "split": split, "group": group, "replicates": len(rows),
                   "positives": rows[0]["positives"], "negatives": rows[0]["negatives"]}
        for name in metric_names:
            summary[f"{name}_mean"], summary[f"{name}_sd"] = mean_sd([row[name] for row in rows])
            if name == "ap" and group == "":
                summary.update(bootstrap_rows[(tool, split)])
        summary_rows.append(summary)

    for name, rows in [("replicate_metrics.csv", replicate_rows), ("summary.csv", summary_rows), ("paired_differences.csv", difference_rows)]:
        fieldnames = list(dict.fromkeys(field for row in rows for field in row))
        with open(out / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {out}/replicate_metrics.csv, summary.csv, paired_differences.csv")


if __name__ == "__main__":
    main()
