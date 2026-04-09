#!/usr/bin/env python3
"""
Produce three outputs from benchmark evaluation results:
  1. metrics_4panel.png  — bar charts: Precision, Recall, Specificity, F1
  2. auc_curves.png      — ROC curves + PR curves for all tools
  3. comparison_table.png — summary table of all metrics per tool
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
DEFAULT_METRICS_CSV = BENCHMARK_DIR / "evaluated" / "balanced_benchmark" / "metrics.csv"
DEFAULT_CURVES_JSON = BENCHMARK_DIR / "evaluated" / "balanced_benchmark" / "curves.json"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "evaluated" / "balanced_benchmark"

BAR_METRICS = [
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", default=str(DEFAULT_METRICS_CSV))
    parser.add_argument("--curves-json", default=str(DEFAULT_CURVES_JSON))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--title", default="Balanced Benchmark Metrics")
    return parser.parse_args()


def load_metrics(path):
    rows = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            parsed = {"tool": row["tool"]}
            for key in ("precision", "recall", "specificity", "accuracy", "f1", "mcc", "roc_auc", "pr_auc"):
                val = row.get(key, "")
                parsed[key] = float(val) if val else None
            for key in ("n", "tp", "fp", "tn", "fn"):
                parsed[key] = int(row[key]) if row.get(key) else 0
            rows.append(parsed)
    return rows


def load_curves(path):
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── helpers ────────────────────────────────────────────────────────────────────

def style_axis(ax):
    ax.set_ylim(0, 1.05)
    ax.set_facecolor(AXIS_FACE_COLOR)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.4, color=GRID_COLOR)
    ax.tick_params(axis="x", labelrotation=35, labelsize=11, length=0)
    ax.tick_params(axis="y", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)


def annotate_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(h + 0.02, 1.01),
            f"{h:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )


# ── plot 1: 4-panel bar chart ──────────────────────────────────────────────────

def plot_bar_metrics(rows, title, out_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    fig.patch.set_facecolor("white")
    axes = axes.flatten()
    x = np.arange(len(rows))

    for ax, (key, label) in zip(axes, BAR_METRICS):
        panel = sorted(rows, key=lambda r: (r[key] or 0), reverse=True)
        vals = [(r[key] or 0) for r in panel]
        labels = [TOOL_LABELS.get(r["tool"], r["tool"]) for r in panel]
        colors = [TOOL_COLORS.get(r["tool"], "#4C72B0") for r in panel]
        bars = ax.bar(x, vals, width=0.72, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=1.5)
        ax.set_title(label, fontsize=14, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, ha="right")
        style_axis(ax)
        annotate_bars(ax, bars)

    fig.suptitle(title, fontsize=18, fontweight="bold", x=0.5, y=1.02)
    fig.tight_layout()
    out = Path(out_dir) / "metrics_4panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot: {out}")


# ── plot 2: ROC + PR AUC curves ───────────────────────────────────────────────

def plot_auc_curves(rows, curves, title, out_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    for ax in (ax_roc, ax_pr):
        ax.set_facecolor(AXIS_FACE_COLOR)
        ax.grid(linestyle="--", linewidth=0.7, alpha=0.4, color=GRID_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ROC panel
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.50)")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.set_xlabel("False Positive Rate", fontsize=13)
    ax_roc.set_ylabel("True Positive Rate", fontsize=13)
    ax_roc.set_title("ROC Curves", fontsize=15, fontweight="bold")

    # PR panel — baseline = fraction of positives
    n_total = sum(r["n"] for r in rows[:1]) if rows else 0
    n_pos = sum(r["tp"] + r["fn"] for r in rows[:1]) if rows else 0
    baseline = n_pos / n_total if n_total else 0.5
    ax_pr.axhline(baseline, color="k", linestyle="--", linewidth=1, alpha=0.5,
                  label=f"Random (AP={baseline:.2f})")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.set_xlabel("Recall", fontsize=13)
    ax_pr.set_ylabel("Precision", fontsize=13)
    ax_pr.set_title("Precision-Recall Curves", fontsize=15, fontweight="bold")

    for row in rows:
        tool = row["tool"]
        color = TOOL_COLORS.get(tool, "#4C72B0")
        label_name = TOOL_LABELS.get(tool, tool)
        curve = curves.get(tool, {})

        roc_auc = row.get("roc_auc")
        pr_auc = row.get("pr_auc")
        fpr = curve.get("fpr")
        tpr = curve.get("tpr")
        prec = curve.get("precision_curve")
        rec = curve.get("recall_curve")

        if fpr and tpr and roc_auc is not None:
            ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                        label=f"{label_name} (AUC={roc_auc:.3f})")
        if prec and rec and pr_auc is not None:
            ax_pr.plot(rec, prec, color=color, linewidth=2,
                       label=f"{label_name} (AP={pr_auc:.3f})")

    ax_roc.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax_pr.legend(loc="upper right", fontsize=10, framealpha=0.9)

    fig.suptitle(title, fontsize=17, fontweight="bold", x=0.5, y=1.02)
    fig.tight_layout()
    out = Path(out_dir) / "auc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot: {out}")


# ── plot 3: comparison table ───────────────────────────────────────────────────

def plot_comparison_table(rows, title, out_dir):
    col_keys = ["tool", "n", "precision", "recall", "specificity", "f1", "mcc", "roc_auc", "pr_auc"]
    col_headers = ["Tool", "N", "Precision", "Recall", "Specificity", "F1", "MCC", "ROC AUC", "PR AUC"]

    # Sort by ROC AUC descending, then F1
    sorted_rows = sorted(rows, key=lambda r: (r.get("roc_auc") or 0, r.get("f1") or 0), reverse=True)

    def fmt(val, key):
        if key == "tool":
            return TOOL_LABELS.get(val, val)
        if key == "n":
            return str(int(val)) if val is not None else "—"
        if val is None:
            return "—"
        return f"{float(val):.4f}"

    cell_data = [[fmt(r.get(k), k) for k in col_keys] for r in sorted_rows]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig_h = max(2.5, 0.5 * len(sorted_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    table = ax.table(
        cellText=cell_data,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header row
    for j in range(len(col_headers)):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row colours; highlight best value per metric column
    metric_cols = {i: k for i, k in enumerate(col_keys) if k not in ("tool", "n")}
    best_vals = {}
    for col_i, key in metric_cols.items():
        vals = [r.get(key) for r in sorted_rows if r.get(key) is not None]
        best_vals[col_i] = max(vals) if vals else None

    for row_i, row in enumerate(sorted_rows):
        bg = "#F0F4F8" if row_i % 2 == 0 else "white"
        tool_color = TOOL_COLORS.get(row["tool"], "#CCCCCC")
        for col_i, key in enumerate(col_keys):
            cell = table[row_i + 1, col_i]
            cell.set_facecolor(bg)
            if col_i == 0:
                cell.set_facecolor(tool_color + "55")  # tinted tool name column
            val = row.get(key)
            if col_i in metric_cols and val is not None and best_vals.get(col_i) is not None:
                if abs(float(val) - best_vals[col_i]) < 1e-9:
                    cell.set_facecolor("#ABEBC6")  # green highlight for best

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout()
    out = Path(out_dir) / "comparison_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot: {out}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    metrics_csv = Path(args.metrics_csv)
    curves_json = Path(args.curves_json)
    out_dir = Path(args.out_dir)

    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_metrics(metrics_csv)
    curves = load_curves(curves_json)

    if not rows:
        raise RuntimeError(f"No metric rows found in {metrics_csv}")

    plot_bar_metrics(rows, args.title, out_dir)
    plot_auc_curves(rows, curves, args.title, out_dir)
    plot_comparison_table(rows, args.title, out_dir)


if __name__ == "__main__":
    main()
