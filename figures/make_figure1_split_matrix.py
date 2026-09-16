#!/usr/bin/env python3
"""Generate the pre-miRBench split-design figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


def labelled_box(axis, x, y, width, height, title, subtitle, color):
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            facecolor="white",
            edgecolor=color,
            linewidth=2,
        )
    )
    axis.text(
        x + width / 2,
        y + height * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=color,
    )
    axis.text(
        x + width / 2,
        y + height * 0.31,
        subtitle,
        ha="center",
        va="center",
        fontsize=11,
        color="#4B5563",
    )


def main():
    figure, axis = plt.subplots(figsize=(11.5, 7.2))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(0.08, 0.89, "Development", fontsize=16, fontweight="bold", color="#263648")
    labelled_box(axis, 0.30, 0.82, 0.27, 0.12, "TRAIN", "model fitting", "#315A84")
    labelled_box(axis, 0.62, 0.82, 0.27, 0.12, "VALIDATION", "model selection", "#49734E")

    axis.plot([0.08, 0.92], [0.755, 0.755], color="#D0D5DB", linewidth=1.2)
    axis.text(0.08, 0.68, "Test sets", fontsize=16, fontweight="bold", color="#263648")

    left = 0.32
    bottom = 0.12
    cell_width = 0.27
    cell_height = 0.20
    column_gap = 0.05
    row_gap = 0.06
    right = left + cell_width + column_gap
    top = bottom + cell_height + row_gap

    axis.text(
        (left + right + cell_width) / 2,
        0.66,
        "miRNA family",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color="#4B5563",
    )
    axis.text(left + cell_width / 2, 0.60, "In training", ha="center", fontsize=12.5, fontweight="bold", color="#263648")
    axis.text(right + cell_width / 2, 0.60, "Held out", ha="center", fontsize=12.5, fontweight="bold", color="#263648")

    axis.text(0.20, 0.60, "Species", ha="center", fontsize=13, fontweight="bold", color="#4B5563")
    axis.text(0.275, top + cell_height / 2, "In training", ha="right", va="center", fontsize=12.5, fontweight="bold", color="#263648")
    axis.text(0.275, bottom + cell_height / 2, "Held out", ha="right", va="center", fontsize=12.5, fontweight="bold", color="#263648")

    cells = (
        (left, top, "TEST 1", "#315A84"),
        (right, top, "TEST 2", "#267484"),
        (left, bottom, "TEST 3", "#95640A"),
        (right, bottom, "TEST 4", "#61439A"),
    )
    for x, y, title, color in cells:
        axis.add_patch(
            Rectangle(
                (x, y),
                cell_width,
                cell_height,
                facecolor="#FAFAFA",
                edgecolor="#7B8490",
                linewidth=1.5,
            )
        )
        axis.text(
            x + cell_width / 2,
            y + cell_height / 2,
            title,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color=color,
        )

    output_dir = Path(__file__).resolve().parent
    figure.savefig(
        output_dir / "figure1_split_matrix.svg",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None},
    )
    figure.savefig(
        output_dir / "figure1_split_matrix.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "pre-miRBench"},
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
