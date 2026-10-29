#!/usr/bin/env python3
"""Genome-scale precision-recall curves on the four held-out genomes, in the style of make_pr_curves_v3.py.

Reads the curve points of genome_pr_points.py (<method>_<genome>.csv: the points where recall rises, which give the AP
exactly). Axes 0 to 1 as in make_pr_curves_v3.py (also 17.5 in wide); the dotted line is the prevalence of annotated
precursors among the candidates (1 in 8,000 to 32,000). Predictors: seed 1, the agent's networks for the pre-miRBench model; retrained MirMachine with cutoffs
fitted on the training genomes; ncOrtho with training-family references (chicken only). A method without points yet
is left out and listed.

Usage: python make_genome_pr_curves_v3.py --points-dir <results>/genome/pr_curves
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
METHODS = {  # points file prefix: (label, colour); colours as in the other v3 figures
    "deepmir": ("DeepMir", "#8C564B"),
    "deepmirgene": ("deepMiRGene", "#0072B2"),
    "dnnpremir": ("dnnPreMiR", "#D55E00"),
    "mirdnn": ("mirDNN", "#CC79A7"),
    "mire2e": ("miRe2e", "#009E73"),
    "mustard": ("MuStARD", "#E69F00"),
    "premirbench_model_v3": ("pre-miRBench", "#332288"),
    "mirmachine_retrained_genome_cutoffs": ("MirMachine 2", "#000000"),
    "ncortho_training_family_refs": ("ncOrtho", "#A6A6A6"),
}
GENOMES = {"dme": "Drosophila melanogaster", "dan": "Drosophila ananassae", "dya": "Drosophila yakuba", "gga": "Gallus gallus"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-dir", required=True)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 18,
        "axes.titlesize": 18, "axes.titleweight": "bold",
        "axes.labelsize": 18, "axes.labelweight": "bold",
        "xtick.labelsize": 18, "ytick.labelsize": 18,
        "legend.fontsize": 18, "svg.fonttype": "none",
    })
    figure, axes = plt.subplots(2, 2, figsize=(17.5, 10), sharex=True, sharey=True)
    missing, drawn = [], {}
    for axis, (genome, title) in zip(axes.flat, GENOMES.items()):
        prevalence = None
        for method, (name, color) in METHODS.items():
            path = Path(args.points_dir) / f"{method}_{genome}.csv"
            if not path.exists():
                if not (method.startswith("ncortho") and genome != "gga"):
                    missing.append(path.name)
                continue
            rows = list(csv.DictReader(open(path, newline="")))
            recall = np.array([float(row["recall"]) for row in rows])
            precision = np.array([float(row["precision"]) for row in rows])
            positives, negatives = int(rows[0]["positives"]), int(rows[0]["negatives"])
            if prevalence is None:
                prevalence = positives / (positives + negatives)
            elif abs(prevalence - positives / (positives + negatives)) > 1e-15:
                raise SystemExit(f"{path}: candidate counts differ from the other methods")
            line, = axis.step(recall, precision, where="post", color=color, linestyle="-",
                              linewidth=3.2 if name == "pre-miRBench" else 2.2, label=name)
            drawn[name] = line
        axis.axhline(prevalence, color="#888888", lw=0.8, ls=":")
        axis.set_title(title, fontstyle="italic")
        axis.set(xlim=(0, 1.01), ylim=(0, 1.015))
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontweight("bold")
    for axis in axes[1]:
        axis.set_xlabel("Recall")
    for axis in axes[:, 0]:
        axis.set_ylabel("Precision")
    names = [name for name, _ in METHODS.values() if name in drawn]
    figure.legend([drawn[name] for name in names], names, loc="lower center", ncol=len(names), frameon=False,
                  handlelength=1.2, handletextpad=0.3, columnspacing=0.6, borderaxespad=0, bbox_to_anchor=(0.5, 0.015))
    figure.tight_layout(rect=(0, 0.085, 1, 1))
    for extension in ("png", "svg"):
        figure.savefig(HERE / f"figure_genome_pr_curves_v3.{extension}", dpi=300)
    plt.close(figure)
    if missing:
        print(f"not drawn (no points yet): {', '.join(missing)}")


if __name__ == "__main__":
    main()
