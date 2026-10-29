#!/usr/bin/env python3
"""Precision–recall curves on the four v3 tests, in the style of the submitted Figure 4 (manuscript/make_figure4.py).

Changes from Figure 4: 17.5 in wide, as the genome curves (make_genome_pr_curves_v3.py), whose nine-entry legend needs it; curves are computed here from the v3 predictions (one model per method, seed 1; the agent's networks for the pre-miRBench
model, as in the genome-scale evaluation), and retrained MirMachine with cutoffs fitted on the training genomes is
added when its predictions exist. The curve points are written to pr_curves_v3.csv; the AP of each curve is checked
against replicate_metrics.csv of the benchmark summary.

Usage: python make_pr_curves_v3.py --predictions-dir <results>/benchmark/predictions --replicate-metrics <results>/benchmark/summary/replicate_metrics.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


HERE = Path(__file__).resolve().parent
METHODS = {  # predictions folder: (label, colour); predictor colours as in Figure 4, MirMachine as in the v2 figures
    "deepmir_v3_rep1": ("DeepMir", "#8C564B"),
    "deepmirgene_v3_rep1": ("deepMiRGene", "#0072B2"),
    "dnnpremir_v3_rep1": ("dnnPreMiR", "#D55E00"),
    "mirdnn_v3_rep1": ("mirDNN", "#CC79A7"),
    "mire2e_v3_rep1": ("miRe2e", "#009E73"),
    "mustard_v3_rep1": ("MuStARD", "#E69F00"),
    "premirbench_model_v3_original": ("pre-miRBench", "#332288"),
    "mirmachine_retrained_genome_cutoffs": ("MirMachine 2", "#000000"),
}
TESTS = {
    "test_known_species_known_family": "Test 1: Known species / known families",
    "test_known_species_heldout_family": "Test 2: Known species / held-out families",
    "test_heldout_species_known_family": "Test 3: Held-out species / known families",
    "test_heldout_species_heldout_family": "Test 4: Held-out species / held-out families",
}


def read_predictions(path):
    by_split = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            labels, scores = by_split.setdefault(row["split"], ([], []))
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
    return {split: (np.array(labels), np.array(scores)) for split, (labels, scores) in by_split.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--replicate-metrics", required=True)
    args = parser.parse_args()

    expected_ap = {}
    with open(args.replicate_metrics, newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["group"]:
                expected_ap[(row["replicate"], row["split"])] = float(row["ap"])
    methods = {run: style for run, style in METHODS.items()
               if (Path(args.predictions_dir) / run / "predictions.csv").exists()}
    missing = sorted(set(METHODS) - set(methods))
    if missing:
        print(f"not drawn (no predictions yet): {', '.join(missing)}")

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 18,
        "axes.titlesize": 18, "axes.titleweight": "bold",
        "axes.labelsize": 18, "axes.labelweight": "bold",
        "xtick.labelsize": 18, "ytick.labelsize": 18,
        "legend.fontsize": 18, "svg.fonttype": "none",
    })
    figure, axes = plt.subplots(2, 2, figsize=(17.5, 10), sharex=True, sharey=True)
    curve_rows = []
    predictions = {run: read_predictions(Path(args.predictions_dir) / run / "predictions.csv") for run in methods}
    for axis, (split, title) in zip(axes.flat, TESTS.items()):
        reference = None
        for run, (name, color) in methods.items():
            labels, scores = predictions[run][split]
            if reference is None:
                reference = labels
            if len(labels) != len(reference) or labels.sum() != reference.sum():
                raise SystemExit(f"{run} {split}: label counts differ from the other methods")
            ap = average_precision_score(labels, scores)
            if (run, split) in expected_ap and abs(ap - expected_ap[(run, split)]) > 1e-12:
                raise SystemExit(f"{run} {split}: AP {ap} differs from replicate_metrics.csv {expected_ap[(run, split)]}")
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            curve_rows += [{"split": split, "method": name, "predictions": run, "threshold": t, "precision": p, "recall": r}
                           for t, p, r in zip(np.r_[thresholds, np.nan], precision, recall)]
            axis.step(recall, precision, where="post", color=color, linestyle="-",
                      linewidth=3.2 if name == "pre-miRBench" else 2.2, label=name)
            print(f"{split} {name}: AP {ap:.4f}")
        axis.axhline(reference.sum() / len(reference), color="#888888", lw=0.8, ls=":")
        axis.set_title(title)
        axis.set(xlim=(0, 1.01), ylim=(0, 1.015))
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontweight("bold")
    for axis in axes[1]:
        axis.set_xlabel("Recall")
    for axis in axes[:, 0]:
        axis.set_ylabel("Precision")
    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=len(names), frameon=False,
                  handlelength=1.2, handletextpad=0.3, columnspacing=0.6,
                  borderaxespad=0, bbox_to_anchor=(0.5, 0.015))
    figure.tight_layout(rect=(0, 0.085, 1, 1))
    for extension in ("png", "svg"):
        figure.savefig(HERE / f"figure_pr_curves_v3.{extension}", dpi=300)
    plt.close(figure)
    with open(HERE / "pr_curves_v3.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)


if __name__ == "__main__":
    main()
