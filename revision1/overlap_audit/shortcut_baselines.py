#!/usr/bin/env python3
"""Simple baselines that test whether pre-miRBench labels are predictable from shortcut cues.

Each baseline is fitted on the training partition only and scored by AP on the four tests.
With --predictions-dir, per-record scores for the validation and test partitions are also written
as <dir>/<slug>/predictions.csv, in the format of pipeline/evaluate.py (input to summarize_benchmark.py).
"""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPLIT_NAME = {
    "test_known_species_known_family": "Test 1",
    "test_known_species_heldout_family": "Test 2",
    "test_heldout_species_known_family": "Test 3",
    "test_heldout_species_heldout_family": "Test 4",
}
SLUG = {
    "MFE only (no fitting; score = -MFE)": "baseline_mfe",
    "Four hairpin-filter features + MFE (LR)": "baseline_hairpin_filter_lr",
    "Mining features, 22 (RF)": "baseline_mining_rf",
    "k-mer composition k=1-4 (LR)": "baseline_kmer_lr",
    "Positional paired profile, 20 bins (LR)": "baseline_paired_profile_lr",
    "Paired or unpaired at each of 200 positions (LR)": "baseline_paired_per_position_lr",
    "Hairpin-loop distance from centre (LR)": "baseline_loop_centrality_lr",
    "Mining features + positional profile + centrality (GB)": "baseline_combined_gb",
}
DINUCLEOTIDES = ["".join(p) for p in product("ACGU", repeat=2)]


def longest_run(text, chars):
    best = current = 0
    for char in text:
        current = current + 1 if char in chars else 0
        best = max(best, current)
    return best


def bulges(structure):
    count, inside = 0, False
    for char in structure:
        if char in "()":
            inside = True
        elif inside:
            count += 1
            inside = False
    return count


def entropy(sequence):
    counts = np.array([sequence.count(b) for b in "ACGU"], dtype=float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def dinucleotide_freq(sequence):
    pairs = [sequence[i:i + 2] for i in range(len(sequence) - 1)]
    total = sum(1 for p in pairs if p in DINUCLEOTIDES) or 1
    return [pairs.count(d) / total for d in DINUCLEOTIDES]


def mining_features(row):
    s, st = row["sequence"], row["structure"]
    paired = (st.count("(") + st.count(")")) / len(st)
    stem = max(longest_run(st, "("), longest_run(st, ")"))
    return [float(row["mfe"]), *dinucleotide_freq(s), stem, longest_run(st, "."), bulges(st), paired, entropy(s)]


def kmer_features(sequence, k_values=(1, 2, 3, 4)):
    features = []
    for k in k_values:
        vocab = {"".join(p): i for i, p in enumerate(product("ACGU", repeat=k))}
        counts = np.zeros(len(vocab))
        for i in range(len(sequence) - k + 1):
            index = vocab.get(sequence[i:i + k])
            if index is not None:
                counts[index] += 1
        features.extend(counts / max(1, counts.sum()))
    return features


def paired_profile(structure, bins=20):
    """Fraction of paired nucleotides in each of `bins` equal-width positional bins of the window."""
    paired = np.array([c != "." for c in structure], dtype=float)
    return [chunk.mean() for chunk in np.array_split(paired, bins)]


def paired_per_position(structure):
    """1 for each paired nucleotide of the window, 0 for each unpaired one (one feature per position)."""
    return [float(c != ".") for c in structure]


def hairpin_centrality(structure):
    """Distance (nt) from window centre to the nearest hairpin loop centre, and that loop's enclosing stem depth."""
    centre = len(structure) / 2
    best = len(structure)
    stack = []
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            j = stack.pop()
            inner = structure[j + 1:i]
            if "(" not in inner and ")" not in inner:
                best = min(best, abs((i + j) / 2 - centre))
    return [best]


def build_features(df):
    sequences = df["sequence"].str.upper()
    return {
        "MFE only (no fitting; score = -MFE)": None,
        "Four hairpin-filter features + MFE (LR)": np.array(
            [[float(m), (st.count("(") + st.count(")")) / len(st), max(longest_run(st, "("), longest_run(st, ")")), longest_run(st, ".")]
             for m, st in zip(df["mfe"], df["structure"])]
        ),
        "Mining features, 22 (RF)": np.array([mining_features(r) for r in df[["sequence", "structure", "mfe"]].to_dict("records")]),
        "k-mer composition k=1-4 (LR)": np.array([kmer_features(s) for s in sequences]),
        "Positional paired profile, 20 bins (LR)": np.array([paired_profile(st) for st in df["structure"]]),
        "Paired or unpaired at each of 200 positions (LR)": np.array([paired_per_position(st) for st in df["structure"]]),
        "Hairpin-loop distance from centre (LR)": np.array([hairpin_centrality(st) for st in df["structure"]]),
    }


def make_model(name):
    if "(RF)" in name:
        return RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42)
    if "(GB)" in name:
        return HistGradientBoostingClassifier(random_state=42)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.dataset, keep_default_na=False)
    df["label"] = df["label"].astype(int)
    df = df[df["split"].isin(["train", "valid", *SPLIT_NAME])].reset_index(drop=True)
    features = build_features(df)
    features["Mining features + positional profile + centrality (GB)"] = np.hstack(
        [features["Mining features, 22 (RF)"], features["Positional paired profile, 20 bins (LR)"], features["Hairpin-loop distance from centre (LR)"]]
    )
    train = (df["split"] == "train").to_numpy()
    y = df["label"].to_numpy()
    rows = []
    for name, x in features.items():
        if x is None:
            score = -df["mfe"].astype(float).to_numpy()
        else:
            model = make_model(name).fit(x[train], y[train])
            score = model.predict_proba(x)[:, 1]
        row = {"baseline": name}
        for split, label in SPLIT_NAME.items():
            mask = (df["split"] == split).to_numpy()
            row[label] = average_precision_score(y[mask], score[mask])
        rows.append(row)
        print(row, flush=True)
        if args.predictions_dir:
            out = Path(args.predictions_dir) / SLUG[name]
            out.mkdir(parents=True, exist_ok=True)
            scored = df.loc[df["split"] != "train", ["split", "record_id", "species", "label"]].copy()
            scored.insert(0, "tool", SLUG[name])
            scored["score"] = score[(df["split"] != "train").to_numpy()]
            scored.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
