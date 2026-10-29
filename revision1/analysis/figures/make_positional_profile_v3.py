#!/usr/bin/env python3
"""Fraction of records with a paired base at each window position, positives against negatives, in the v3 benchmark.
Style as Figure 2 (manuscript/make_hairpin_figure.py): Arial 18, the same class colours, legend on one line above the
panel. Single-panel version of make_positional_profile_v1_v3.py without the v1 comparison (user, 25 Oct).

Usage: python make_positional_profile_v3.py <v3 dataset.csv>
Writes figure_positional_profile_v3.{png,svg} and positional_profile_v3.csv next to this script.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "Arial", "font.size": 18, "axes.labelsize": 18, "axes.labelweight": "bold",
                     "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18, "svg.fonttype": "none"})

HERE = Path(__file__).resolve().parent
COLORS = {"Positive": "#D55E00", "Negative": "#0072B2"}
WINDOW = 200


def paired_profile(dataset):
    paired = {"Positive": [], "Negative": []}
    with open(dataset, newline="") as handle:
        for row in csv.DictReader(handle):
            if len(row["structure"]) != WINDOW:
                raise SystemExit(f"{dataset}: {row['record_id']} has a {len(row['structure'])}-nt structure")
            paired["Positive" if row["label"] == "1" else "Negative"].append(
                np.frombuffer(row["structure"].encode(), dtype=np.uint8) != ord("."))
    return {name: (np.mean(rows, axis=0), len(rows)) for name, rows in paired.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    profiles = paired_profile(args.dataset)
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    positions = np.arange(1, WINDOW + 1)
    for name in ("Negative", "Positive"):
        axis.plot(positions, profiles[name][0], color=COLORS[name], linewidth=1.8)
    axis.set_xlabel("Position in window (nt)")
    axis.set_ylabel("Fraction of windows\nwith a paired base")  # my wording, approved by the user on 25 Oct
    axis.set(xlim=(1, WINDOW), ylim=(0, 1))
    axis.spines[["top", "right"]].set_visible(False)
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontweight("bold")
    handles = [Line2D([0], [0], color=COLORS[name], linewidth=1.8, label=name) for name in ("Negative", "Positive")]
    figure.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    figure.savefig(HERE / "figure_positional_profile_v3.png", dpi=300)
    figure.savefig(HERE / "figure_positional_profile_v3.svg")
    plt.close(figure)

    with open(HERE / "positional_profile_v3.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "records", "position", "fraction_paired"])
        for name, (profile, count) in profiles.items():
            writer.writerows([name.lower(), count, position, f"{value:.6f}"] for position, value in zip(positions, profile))
    print({name: count for name, (_, count) in profiles.items()})


if __name__ == "__main__":
    main()
