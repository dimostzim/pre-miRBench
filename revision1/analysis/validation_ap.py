#!/usr/bin/env python3
"""Validation-partition AP and AUROC of every model in results/benchmark/predictions, with the functions of
summarize_benchmark.py (tied scores as one threshold). summarize_benchmark.py reports the four tests only.
usage: validation_ap.py <repo> <results dir>"""
import csv
import sys
from pathlib import Path

import numpy as np

repo, results = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / "pipeline"))
from summarize_benchmark import auroc, average_precision_rows  # noqa: E402

out = results / "benchmark" / "validation_ap.csv"
if out.exists():
    raise SystemExit(f"{out} exists")
rows = []
for path in sorted((results / "benchmark" / "predictions").glob("*/predictions.csv")):
    records = [row for row in csv.DictReader(open(path)) if row["split"] == "valid"]
    labels = np.array([int(row["label"]) for row in records])
    scores = np.array([float(row["score"]) for row in records])
    rows.append({"tool": records[0]["tool"], "replicate": path.parent.name, "positives": int(labels.sum()),
                 "negatives": int((labels == 0).sum()), "ap": float(average_precision_rows(labels, scores[None, :])[0]),
                 "auroc": float(auroc(labels, scores))})
with open(out, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"{len(rows)} models -> {out}")
