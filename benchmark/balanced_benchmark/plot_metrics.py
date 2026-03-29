#!/usr/bin/env python3
"""
Create a 4-panel balanced benchmark bar plot.

Panels:
- Precision
- Recall
- Specificity
- F1

Each panel shows one bar per tool from benchmark/evaluated/balanced_benchmark/metrics.csv.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
DEFAULT_METRICS_CSV = BENCHMARK_DIR / "evaluated" / "balanced_benchmark" / "metrics.csv"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "evaluated" / "balanced_benchmark"

METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("specificity", "Specificity"),
    ("f1", "F1"),
]

TOOL_LABELS = {
    "deepmir": "DeepMir",
    "deepmirgene": "deepMiRGene",
    "dnnpremir": "DNNPreMir",
    "mirdnn": "mirDNN",
    "mire2e": "miRe2e",
    "mustard": "MuStARD",
}

TOOL_COLORS = {
    "deepmir": "#5AA9D6",
    "deepmirgene": "#C46A9B",
    "dnnpremir": "#E9A43B",
    "mirdnn": "#6FB38D",
    "mire2e": "#8F92E8",
    "mustard": "#E07A3F",
}

BAR_EDGE_COLOR = "#111111"
SPINE_COLOR = "#111111"
GRID_COLOR = "#B7C0CC"
AXIS_FACE_COLOR = "#FCFCFD"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 4-panel balanced benchmark metrics (precision, recall, specificity, F1)."
    )
    parser.add_argument(
        "--metrics-csv",
        default=str(DEFAULT_METRICS_CSV),
        help=f"Metrics CSV from benchmark/balanced_benchmark/evaluate_outputs.py (default: {DEFAULT_METRICS_CSV})",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for the plot (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-name",
        default="metrics_4panel",
        help='Output filename stem without extension (default: "metrics_4panel")',
    )
    parser.add_argument(
        "--title",
        default="Balanced Benchmark Metrics",
        help='Figure title (default: "Balanced Benchmark Metrics")',
    )
    return parser.parse_args()


def load_metrics(path: Path):
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {"tool": row["tool"]}
            for key, _ in METRICS:
                parsed[key] = float(row[key])
            rows.append(parsed)
    return rows


def style_axis(ax):
    ax.set_ylim(0, 1.0)
    ax.set_facecolor(AXIS_FACE_COLOR)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.4, color=GRID_COLOR)
    ax.tick_params(axis="x", labelrotation=35, labelsize=12, length=0)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(height + 0.02, 0.98),
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )


def plot_metrics(rows, title: str, output_path: Path):
    tools = [row["tool"] for row in rows]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    fig.patch.set_facecolor("white")
    axes = axes.flatten()
    x_positions = np.arange(len(rows))
    width = 0.72

    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        panel_rows = sorted(rows, key=lambda row: row[metric_key], reverse=True)
        values = [row[metric_key] for row in panel_rows]
        labels = [TOOL_LABELS.get(row["tool"], row["tool"]) for row in panel_rows]
        colors = [TOOL_COLORS.get(row["tool"], "#4C72B0") for row in panel_rows]
        bars = ax.bar(
            x_positions,
            values,
            width=width,
            color=colors,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=1.8,
        )
        ax.set_title(metric_title, fontsize=16, fontweight="bold", loc="center", pad=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, ha="right")
        style_axis(ax)
        annotate_bars(ax, bars)

    fig.suptitle(title, fontsize=21, fontweight="bold", x=0.5, y=1.025, ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    metrics_csv = Path(args.metrics_csv)
    out_dir = Path(args.out_dir)

    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_metrics(metrics_csv)
    if not rows:
        raise RuntimeError(f"No metric rows found in {metrics_csv}")

    output_path = out_dir / f"{args.output_name}.png"
    plot_metrics(rows, args.title, output_path)
    print(f"plot: {output_path}")


if __name__ == "__main__":
    main()
