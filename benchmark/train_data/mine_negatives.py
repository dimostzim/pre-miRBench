#!/usr/bin/env python3
"""
Iteratively mine hard negatives from a hairpin-like negative pool.

The mining model is intentionally lightweight: an ensemble of random forests
trained on sequence, structure, and MFE features. Rows that the ensemble
systematically scores as pre-miRNA-like are selected as hard negatives.
"""
import argparse
import csv
import math
import os
import random

import numpy as np


DINUCLEOTIDES = [left + right for left in "ACGU" for right in "ACGU"]


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, extra_fields=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    base_fields = [
        "window_id",
        "chrom",
        "start",
        "end",
        "strand",
        "sequence",
        "structure",
        "mfe",
        "mirna_id",
        "label",
    ]
    fields = list(base_fields)
    for field in extra_fields or []:
        if field not in fields:
            fields.append(field)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dinucleotide_freq(sequence):
    sequence = sequence.upper().replace("T", "U")
    counts = {dinuc: 0 for dinuc in DINUCLEOTIDES}
    total = 0
    for index in range(len(sequence) - 1):
        dinuc = sequence[index:index + 2]
        if dinuc in counts:
            counts[dinuc] += 1
            total += 1
    if total == 0:
        return [0.0 for _ in DINUCLEOTIDES]
    return [counts[dinuc] / total for dinuc in DINUCLEOTIDES]


def sequence_entropy(sequence):
    sequence = sequence.upper().replace("T", "U")
    counts = {base: 0 for base in "ACGU"}
    total = 0
    for base in sequence:
        if base in counts:
            counts[base] += 1
            total += 1
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count:
            prob = count / total
            entropy -= prob * math.log2(prob)
    return entropy


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


def structure_features(structure):
    if not structure:
        return [0.0, 0.0, 0.0, 0.0]
    paired = structure.count("(") + structure.count(")")
    paired_frac = paired / len(structure)
    max_stem = max(longest_run(structure, "("), longest_run(structure, ")"))
    max_loop = longest_run(structure, ".")
    bulges = 0
    in_paired_region = False
    for char in structure:
        if char in "()":
            in_paired_region = True
        elif char == "." and in_paired_region:
            bulges += 1
            in_paired_region = False
    return [float(max_stem), float(max_loop), float(bulges), paired_frac]


def features(row):
    return [
        float(row["mfe"]),
        *dinucleotide_freq(row["sequence"]),
        *structure_features(row["structure"]),
        sequence_entropy(row["sequence"]),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Mine hard negatives from a folded hairpin pool.")
    parser.add_argument("--positives", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--hard-negatives", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--ratio", type=float, default=5.0, help="Negative:positive ratio used while training mining rounds.")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=10)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--consensus", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise SystemExit("scikit-learn is required. Use the benchmark conda environment.") from exc

    positives = read_rows(args.positives)
    pool = read_rows(args.pool)
    if not positives:
        raise SystemExit("No positives provided.")
    if not pool:
        raise SystemExit("No negative pool rows provided.")

    rng = random.Random(args.seed)
    positive_features = np.array([features(row) for row in positives], dtype=float)
    pool_features = np.array([features(row) for row in pool], dtype=float)

    target_train_negatives = min(len(pool), max(1, math.ceil(len(positives) * args.ratio)))
    selected = set()
    round_first_seen = {}
    last_scores = np.zeros(len(pool), dtype=float)
    last_consensus = np.zeros(len(pool), dtype=float)

    all_indices = list(range(len(pool)))
    for round_index in range(1, args.rounds + 1):
        selected_indices = sorted(selected)
        remaining = [index for index in all_indices if index not in selected]
        fill_count = max(0, target_train_negatives - len(selected_indices))
        fill_indices = rng.sample(remaining, min(fill_count, len(remaining))) if remaining else []
        train_indices = selected_indices + fill_indices

        x_train = np.vstack([positive_features, pool_features[train_indices]])
        y_train = np.array([1] * len(positive_features) + [0] * len(train_indices))

        member_scores = []
        for member in range(args.ensemble_size):
            classifier = RandomForestClassifier(
                n_estimators=args.trees,
                random_state=args.seed + round_index * 1000 + member,
                class_weight="balanced",
                n_jobs=-1,
            )
            classifier.fit(x_train, y_train)
            member_scores.append(classifier.predict_proba(pool_features)[:, 1])

        score_matrix = np.vstack(member_scores)
        last_scores = score_matrix.mean(axis=0)
        last_consensus = (score_matrix >= args.score_threshold).mean(axis=0)

        new_hard = []
        for index, consensus_value in enumerate(last_consensus):
            if index in selected:
                continue
            if consensus_value >= args.consensus:
                selected.add(index)
                round_first_seen[index] = round_index
                new_hard.append(index)

        print(
            f"round {round_index}: train_negatives={len(train_indices)} "
            f"new_hard={len(new_hard)} total_hard={len(selected)}"
        )
        if not new_hard:
            break

    scored_rows = []
    for index, row in enumerate(pool):
        row_copy = row.copy()
        row_copy["label"] = "0"
        row_copy["score"] = f"{last_scores[index]:.8f}"
        row_copy["consensus"] = f"{last_consensus[index]:.8f}"
        row_copy["hard_round"] = str(round_first_seen.get(index, ""))
        scored_rows.append(row_copy)
    scored_rows.sort(key=lambda item: float(item["score"]), reverse=True)

    hard_rows = [row for row in scored_rows if row["hard_round"]]
    hard_rows.sort(key=lambda item: (int(item["hard_round"]), -float(item["score"])))

    write_rows(args.scores, scored_rows, ["score", "consensus", "hard_round"])
    write_rows(args.hard_negatives, hard_rows, ["score", "consensus", "hard_round"])
    print(f"hard negatives: {len(hard_rows)} -> {args.hard_negatives}")
    print(f"all scored negatives: {len(scored_rows)} -> {args.scores}")


if __name__ == "__main__":
    main()
