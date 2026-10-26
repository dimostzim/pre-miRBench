import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "genome_scale"))

import genome_scale_metrics as gsm  # noqa: E402


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def test_validation_thresholds_calls_and_top_n_recall(tmp_path, monkeypatch):
    candidate = {"chrom": "chr1"}
    candidates = [
        {**candidate, "record_id": "p1", "strand": "+", "hairpin_start": 100, "hairpin_end": 150, "genome_label": "positive"},
        {**candidate, "record_id": "n1", "strand": "+", "hairpin_start": 1000, "hairpin_end": 1050, "genome_label": "negative"},
        {**candidate, "record_id": "n2", "strand": "-", "hairpin_start": 1000, "hairpin_end": 1050, "genome_label": "negative"},
        {**candidate, "record_id": "n3", "strand": "+", "hairpin_start": 5000, "hairpin_end": 5050, "genome_label": "negative"},
        {**candidate, "record_id": "a1", "strand": "+", "hairpin_start": 200, "hairpin_end": 250, "genome_label": "ambiguous"},
    ]
    loci = [{"chrom": "chr1", "start": 100, "end": 160, "name": "x", "strand": "+", "family_id": "A", "masked": 0, "matched_record_id": "p1"}]
    genome = [{"tool": "t", "record_id": r, "score": s} for r, s in [("p1", 0.9), ("n1", 0.9), ("n2", 0.7), ("n3", 0.1), ("a1", 0.95)]]
    # validation F1 is 1 at threshold 0.6, which also is the highest threshold reaching 90% recall
    validation = [{"tool": "t", "split": "valid", "label": y, "score": s} for y, s in [(1, 0.9), (1, 0.6), (0, 0.5), (0, 0.2)]]
    validation.append({"tool": "t", "split": "test_known_species_known_family", "label": 0, "score": 0.99})
    training = [{"split": "train", "label": 1, "family_id": "A"}]
    monkeypatch.setattr(sys, "argv", [
        "genome_scale_metrics.py", "--candidates", write_csv(tmp_path / "c.csv", candidates), "--loci", write_csv(tmp_path / "l.csv", loci),
        "--genome-predictions", write_csv(tmp_path / "g.csv", genome), "--validation-predictions", write_csv(tmp_path / "v.csv", validation),
        "--training-families", write_csv(tmp_path / "d.csv", training), "--genome-bases", "1000000",
        "--output", str(tmp_path / "out.csv"),
    ])
    gsm.main()

    rows = {r["threshold_rule"]: r for r in csv.DictReader(open(tmp_path / "out.csv"))}
    row = rows["valid_max_f1"]
    assert float(row["threshold"]) == 0.6 == float(rows["valid_recall_90"]["threshold"])
    assert float(row["calls_per_mb"]) == 4.0  # p1, n1, n2 and the ambiguous a1
    assert float(row["precision_unambiguous_calls"]) == pytest.approx(1 / 3)
    assert float(row["all_loci_recall"]) == 1.0
    # one locus: the top-1 cutoff falls in the tie of p1 and n1 (the ambiguous a1 is not ranked), the top 2 hold both
    assert (float(row["recall_top_1x_loci"]), float(row["recall_top_2x_loci"])) == (0.5, 1.0)
