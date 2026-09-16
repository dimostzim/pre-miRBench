#!/usr/bin/env python3
"""Plot RNAfold feature distributions for the final benchmark records."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FEATURES = (
    ("mfe", "Minimum free energy (kcal/mol)", -10.0, "le"),
    ("paired_fraction", "Paired-nucleotide fraction", 0.40, "ge"),
    ("max_stem", "Maximum contiguous stem (nt)", 8.0, "ge"),
    ("max_unpaired_run", "Maximum unpaired run (nt)", 25.0, "le"),
)


def longest_run(structure, characters):
    longest = 0
    current = 0
    for character in structure:
        if character in characters:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def structure_features(structure):
    paired = structure.count("(") + structure.count(")")
    return {
        "paired_fraction": paired / len(structure),
        "max_stem": max(longest_run(structure, "("), longest_run(structure, ")")),
        "max_unpaired_run": longest_run(structure, "."),
    }


def read_features(dataset_path):
    values = {
        "Positive": {feature[0]: [] for feature in FEATURES},
        "Negative": {feature[0]: [] for feature in FEATURES},
    }
    with dataset_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            record_class = "Positive" if row["label"] == "1" else "Negative"
            derived = structure_features(row["structure"])
            values[record_class]["mfe"].append(float(row["mfe"]))
            for name in ("paired_fraction", "max_stem", "max_unpaired_run"):
                values[record_class][name].append(derived[name])
    return values


def passes_threshold(values, threshold, direction):
    if direction == "le":
        return values <= threshold
    return values >= threshold


def write_summary(values, output_path):
    fields = ["class", "feature", "n", "minimum", "q25", "median", "q75", "maximum", "mean", "threshold", "pass_count", "pass_fraction"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for record_class in ("Positive", "Negative"):
            for name, _, threshold, direction in FEATURES:
                feature_values = np.asarray(values[record_class][name], dtype=float)
                passed = passes_threshold(feature_values, threshold, direction)
                writer.writerow(
                    {
                        "class": record_class.lower(),
                        "feature": name,
                        "n": len(feature_values),
                        "minimum": f"{feature_values.min():.6g}",
                        "q25": f"{np.quantile(feature_values, 0.25):.6g}",
                        "median": f"{np.median(feature_values):.6g}",
                        "q75": f"{np.quantile(feature_values, 0.75):.6g}",
                        "maximum": f"{feature_values.max():.6g}",
                        "mean": f"{feature_values.mean():.6g}",
                        "threshold": f"{threshold:.6g}",
                        "pass_count": int(passed.sum()),
                        "pass_fraction": f"{passed.mean():.6f}",
                    }
                )


def plot_distributions(values, output_path):
    colors = {"Positive": "#D55E00", "Negative": "#0072B2"}
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.4), constrained_layout=True)

    for axis, (name, label, threshold, direction) in zip(axes.flat, FEATURES):
        combined = np.concatenate(
            [np.asarray(values[record_class][name], dtype=float) for record_class in ("Positive", "Negative")]
        )
        if name in {"max_stem", "max_unpaired_run"}:
            bins = np.arange(np.floor(combined.min()) - 0.5, np.ceil(combined.max()) + 1.5)
        else:
            bins = np.linspace(combined.min(), combined.max(), 55)

        for record_class in ("Negative", "Positive"):
            axis.hist(
                values[record_class][name],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colors[record_class],
                label=f"{record_class} (n={len(values[record_class][name]):,})",
            )

        relation = "≤" if direction == "le" else "≥"
        axis.axvline(threshold, color="#333333", linestyle="--", linewidth=1.3, label=f"Filter: {relation} {threshold:g}")
        axis.set_xlabel(label)
        axis.set_ylabel("Density")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)

    figure.suptitle("RNAfold features in the final pre-miRBench records", fontsize=13)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Canonical pre-miRBench dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    values = read_features(args.dataset)
    write_summary(values, args.output_dir / "hairpin_feature_summary.tsv")
    plot_distributions(values, args.output_dir / "hairpin_feature_distributions.png")


if __name__ == "__main__":
    main()
