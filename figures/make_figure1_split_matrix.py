#!/usr/bin/env python3
"""Generate the pre-miRBench split-design figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PARTITIONS = {
    "Train": (69, 687, 4765, 47650),
    "Validation": (61, 171, 631, 6310),
    "Test 1": (62, 177, 707, 7070),
    "Test 2": (61, 59, 677, 6770),
    "Test 3": (2, 112, 207, 2070),
    "Test 4": (2, 59, 69, 690),
}


def add_box(axis, x, y, width, height, title, subtitle, values, face, edge):
    species, families, positives, negatives = values
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.009,rounding_size=0.012",
            facecolor=face,
            edgecolor=edge,
            linewidth=2.2,
        )
    )
    center = x + width / 2
    axis.text(center, y + height * 0.72, title.upper(), ha="center", va="center", fontsize=17, fontweight="bold", color=edge)
    axis.text(center, y + height * 0.52, subtitle, ha="center", va="center", fontsize=12.5, color="#263648")
    axis.text(center, y + height * 0.31, f"{species} species  |  {families} families", ha="center", va="center", fontsize=12, color="#263648")
    axis.text(center, y + height * 0.13, f"{positives:,} positive  |  {negatives:,} negative", ha="center", va="center", fontsize=12, color="#263648")


def main():
    assert sum(row[2] for row in PARTITIONS.values()) == 7056
    assert sum(row[3] for row in PARTITIONS.values()) == 70560

    figure, axis = plt.subplots(figsize=(14.8, 9.6))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    left = 0.27
    width = 0.335
    gap = 0.035
    right = left + width + gap

    axis.text(left, 0.955, "Development partitions", fontsize=19, fontweight="bold", color="#17365D")
    axis.text(left, 0.925, "Used for model fitting and selection; shown separately from the test matrix", fontsize=11.5, color="#526274")
    add_box(axis, left, 0.745, width, 0.15, "Train", "Model fitting", PARTITIONS["Train"], "#E6EEF8", "#244A73")
    add_box(axis, right, 0.745, width, 0.15, "Validation", "Model selection", PARTITIONS["Validation"], "#EAF4E7", "#3B6B43")

    axis.text(left, 0.675, "Test partitions", fontsize=19, fontweight="bold", color="#17365D")
    axis.text(left, 0.642, "A 2 × 2 design defined relative to the training partition", fontsize=11.5, color="#526274")

    axis.text(0.635, 0.595, "miRNA-family status", ha="center", fontsize=14, fontweight="bold", color="#263648")
    axis.text(left + width / 2, 0.557, "Represented in training", ha="center", fontsize=13, fontweight="bold", color="#263648")
    axis.text(right + width / 2, 0.557, "Held out from training", ha="center", fontsize=13, fontweight="bold", color="#263648")

    axis.text(0.125, 0.595, "Species status", ha="center", fontsize=14, fontweight="bold", color="#263648")
    axis.text(0.245, 0.425, "Represented\nin training", ha="right", va="center", fontsize=13, fontweight="bold", color="#263648", linespacing=1.3)
    axis.text(0.245, 0.205, "Held out\nfrom training", ha="right", va="center", fontsize=13, fontweight="bold", color="#263648", linespacing=1.3)

    cell_height = 0.18
    add_box(axis, left, 0.335, width, cell_height, "Test 1", "Known species + known family", PARTITIONS["Test 1"], "#E5F4E9", "#2F6B48")
    add_box(axis, right, 0.335, width, cell_height, "Test 2", "Known species + held-out family", PARTITIONS["Test 2"], "#E4F3F6", "#267484")
    add_box(axis, left, 0.115, width, cell_height, "Test 3", "Held-out species + known family", PARTITIONS["Test 3"], "#FFF2CE", "#95640A")
    add_box(axis, right, 0.115, width, cell_height, "Test 4", "Held-out species + held-out family", PARTITIONS["Test 4"], "#EEE8F8", "#61439A")

    axis.text(
        0.5,
        0.045,
        "All six partitions are record-disjoint   •   Species and family status are defined relative to training   •   Positive:negative ratio = 1:10",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#526274",
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
