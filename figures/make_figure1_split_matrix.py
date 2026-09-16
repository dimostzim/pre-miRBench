#!/usr/bin/env python3
"""Create Figure 1: dataset partitions by overlap with training."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams["svg.hashsalt"] = "premirbench-figure1"


def partition_box(axis, x, y, width, height, title, color, subtitle=None):
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.006",
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
        )
    )
    title_y = y + height * (0.62 if subtitle else 0.50)
    axis.text(
        x + width / 2,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=color,
    )
    if subtitle:
        axis.text(
            x + width / 2,
            y + height * 0.25,
            subtitle,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#4B5563",
        )


def main():
    figure, axis = plt.subplots(figsize=(11.5, 7.2))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.08,
        0.93,
        "Overlap with the training set",
        fontsize=17,
        fontweight="bold",
        color="#263648",
    )

    left = 0.30
    bottom = 0.12
    cell_width = 0.30
    cell_height = 0.29
    gap = 0.035
    right = left + cell_width + gap
    top = bottom + cell_height + gap

    axis.text(
        left + cell_width + gap / 2,
        0.84,
        "miRNA family",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color="#4B5563",
    )
    axis.text(
        left + cell_width / 2,
        0.785,
        "Represented in training",
        ha="center",
        fontsize=11.5,
        fontweight="bold",
        color="#263648",
    )
    axis.text(
        right + cell_width / 2,
        0.785,
        "Held out",
        ha="center",
        fontsize=11.5,
        fontweight="bold",
        color="#263648",
    )

    axis.text(
        0.17,
        0.755,
        "Species",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color="#4B5563",
    )
    axis.text(
        0.265,
        top + cell_height / 2,
        "Represented\nin training",
        ha="right",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#263648",
        linespacing=1.25,
    )
    axis.text(
        0.265,
        bottom + cell_height / 2,
        "Held out",
        ha="right",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#263648",
    )

    for x, y in ((left, top), (right, top), (left, bottom), (right, bottom)):
        axis.add_patch(
            Rectangle(
                (x, y),
                cell_width,
                cell_height,
                facecolor="#FAFAFA",
                edgecolor="#9AA1AA",
                linewidth=1.4,
            )
        )

    inner_x = left + 0.025
    inner_width = cell_width - 0.05
    box_height = 0.065
    partition_box(axis, inner_x, top + 0.195, inner_width, box_height, "TRAIN", "#315A84", "model fitting")
    partition_box(axis, inner_x, top + 0.110, inner_width, box_height, "VALIDATION", "#6B7280", "model selection")
    partition_box(axis, inner_x, top + 0.025, inner_width, box_height, "TEST 1", "#2F6B48")

    partition_box(axis, right + 0.04, top + 0.095, cell_width - 0.08, 0.10, "TEST 2", "#267484")
    partition_box(axis, left + 0.04, bottom + 0.095, cell_width - 0.08, 0.10, "TEST 3", "#95640A")
    partition_box(axis, right + 0.04, bottom + 0.095, cell_width - 0.08, 0.10, "TEST 4", "#61439A")

    axis.text(
        0.5,
        0.055,
        "Each box is a separate set of records",
        ha="center",
        fontsize=10.5,
        color="#4B5563",
    )

    output_dir = Path(__file__).resolve().parent
    figure.savefig(
        output_dir / "figure1_split_matrix.svg",
        format="svg",
        bbox_inches="tight",
        pad_inches=0.12,
        metadata={"Date": None, "Creator": None},
    )
    figure.savefig(
        output_dir / "figure1_split_matrix.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.12,
        metadata={"Software": None},
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
