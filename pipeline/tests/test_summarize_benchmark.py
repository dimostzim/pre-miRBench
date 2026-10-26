import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import summarize_benchmark as sb  # noqa: E402


def test_average_precision_rows_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    labels = (rng.random(300) < 0.1).astype(int)
    labels[0] = 1
    # rounded scores create ties within and across classes
    scores = np.round(rng.random((4, 300)) + labels * rng.random((4, 1)), 1)
    expected = [average_precision_score(labels, row) for row in scores]
    assert sb.average_precision_rows(labels, scores) == pytest.approx(expected)
    assert sb.auroc(labels, scores[0]) == pytest.approx(roc_auc_score(labels, scores[0]))


def write_predictions(path, tool, rows):
    path.parent.mkdir(parents=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tool", "split", "record_id", "species", "label", "score"])
        for split, record_id, species, label, score in rows:
            writer.writerow([tool, split, record_id, species, label, score])


def test_replicate_summary_thresholds_and_paired_differences(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    splits = ["valid", *sb.TEST_SPLITS]
    records = {split: [(f"{'gga' if i % 2 else 'dme'}_{split}_{i}", int(i < 10)) for i in range(60)] for split in splits}
    paths = []
    for tool, replicate, signal in [("good", "rep1", 2.0), ("good", "rep2", 1.5), ("weak", "rep1", 0.3)]:
        rows = [(split, rid, rid.split("_")[0], label, float(label * signal + rng.random()))
                for split in splits for rid, label in records[split]]
        path = tmp_path / f"{tool}_{replicate}" / "predictions.csv"
        write_predictions(path, tool, rows)
        paths.append(str(path))
    monkeypatch.setattr(sys, "argv", ["summarize_benchmark.py", "--predictions", *paths,
                                      "--output-dir", str(tmp_path / "out"), "--bootstrap", "200"])
    sb.main()

    replicate = list(csv.DictReader(open(tmp_path / "out" / "replicate_metrics.csv")))
    summary = {(r["tool"], r["split"], r["group"]): r for r in csv.DictReader(open(tmp_path / "out" / "summary.csv"))}
    split = "test_known_species_known_family"
    good = [float(r["ap"]) for r in replicate if r["tool"] == "good" and r["split"] == split and r["group"] == ""]
    row = summary[("good", split, "")]
    assert float(row["ap_mean"]) == pytest.approx(np.mean(good))
    assert float(row["ap_sd"]) == pytest.approx(np.std(good, ddof=1))
    assert float(row["ap_ci_low"]) <= float(row["ap_mean"]) <= float(row["ap_ci_high"])
    assert summary[("weak", split, "")]["ap_sd"] == "nan"  # one replicate
    # per-species groups only for the held-out-species tests
    assert ("good", "test_heldout_species_known_family", "gga") in summary
    assert ("good", split, "gga") not in summary
    # thresholds come from each replicate's own validation predictions, not from the test split
    weak_valid = [r for r in csv.DictReader(open(paths[2])) if r["split"] == "valid"]
    expected = sb.validation_thresholds(np.array([int(r["label"]) for r in weak_valid]), np.array([float(r["score"]) for r in weak_valid]))
    used = [float(r["valid_max_f1_threshold"]) for r in replicate if r["tool"] == "weak"]
    assert used == pytest.approx([expected["valid_max_f1"]] * len(used))
    difference = next(r for r in csv.DictReader(open(tmp_path / "out" / "paired_differences.csv")) if r["split"] == split)
    assert (difference["tool_a"], difference["tool_b"]) == ("good", "weak")
    assert float(difference["ap_difference"]) == pytest.approx(float(row["ap_mean"]) - float(summary[("weak", split, "")]["ap_mean"]))


def test_mismatched_record_sets_are_rejected(tmp_path, monkeypatch):
    paths = []
    for tool, drop in [("a", False), ("b", True)]:
        rows = [(split, f"r{i}", "x", int(i < 2), float(i)) for split in ["valid", *sb.TEST_SPLITS] for i in range(8)
                if not (drop and split == sb.TEST_SPLITS[0] and i == 7)]
        path = tmp_path / f"{tool}_rep1" / "predictions.csv"
        write_predictions(path, tool, rows)
        paths.append(str(path))
    monkeypatch.setattr(sys, "argv", ["summarize_benchmark.py", "--predictions", *paths, "--output-dir", str(tmp_path / "out"), "--bootstrap", "10"])
    with pytest.raises(ValueError, match="different record set"):
        sb.main()
