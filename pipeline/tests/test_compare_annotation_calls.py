import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comparators"))

import compare_annotation_calls as cac  # noqa: E402


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gff_line(start, end, strand, score):
    # GFF coordinates are 1-based inclusive
    return f"chr1\tcmsearch\thit\t{start}\t{end}\t{score}\t{strand}\t.\t.\n"


def test_locus_groups_and_candidate_metrics(tmp_path, monkeypatch):
    locus = {"chrom": "chr1", "name": "", "matched_record_id": ""}
    write_csv(tmp_path / "loci.csv", [
        {**locus, "start": 100, "end": 160, "strand": "+", "family_id": "A", "masked": 0},
        {**locus, "start": 500, "end": 560, "strand": "-", "family_id": "B", "masked": 0},
        {**locus, "start": 900, "end": 960, "strand": "+", "family_id": "A", "masked": 1},
    ])
    # only positive training rows define training families
    write_csv(tmp_path / "dataset.csv", [
        {"split": "train", "label": 1, "family_id": "A"},
        {"split": "train", "label": 0, "family_id": "B"},
        {"split": "test_known_species_heldout_family", "label": 1, "family_id": "B"},
    ])
    hits = {"locus1": gff_line(101, 160, "+", 50), "wrong_strand": gff_line(501, 560, "+", 40),
            "masked_locus": gff_line(901, 960, "+", 30), "no_locus": gff_line(2001, 2060, "+", 20)}
    (tmp_path / "all.gff").write_text("".join(hits.values()))
    (tmp_path / "called.gff").write_text(hits["locus1"] + hits["masked_locus"] + hits["no_locus"])
    candidate = {"chrom": "chr1", "split": "genome_x", "species": "x"}
    write_csv(tmp_path / "candidates.csv", [
        {**candidate, "record_id": "c1", "strand": "+", "hairpin_start": 105, "hairpin_end": 155, "genome_label": "positive"},
        {**candidate, "record_id": "c2", "strand": "-", "hairpin_start": 505, "hairpin_end": 555, "genome_label": "positive"},
        {**candidate, "record_id": "c3", "strand": "+", "hairpin_start": 2010, "hairpin_end": 2050, "genome_label": "negative"},
        {**candidate, "record_id": "c4", "strand": "+", "hairpin_start": 3000, "hairpin_end": 3050, "genome_label": "negative"},
        {**candidate, "record_id": "c5", "strand": "+", "hairpin_start": 905, "hairpin_end": 955, "genome_label": "ambiguous"},
    ])
    monkeypatch.setattr(sys, "argv", [
        "compare_annotation_calls.py", "--tool", "t", "--format", "mirmachine_gff",
        "--calls", str(tmp_path / "all.gff"), "--called", str(tmp_path / "called.gff"),
        "--loci", str(tmp_path / "loci.csv"), "--training-families", str(tmp_path / "dataset.csv"),
        "--genome-bases", "1000000", "--candidates", str(tmp_path / "candidates.csv"),
        "--summary", str(tmp_path / "summary.csv"), "--predictions", str(tmp_path / "predictions.csv"),
    ])
    cac.main()

    summary = next(csv.DictReader(open(tmp_path / "summary.csv")))
    assert float(summary["calls_per_mb"]) == 3.0
    assert float(summary["precision_calls_on_annotated_loci"]) == pytest.approx(2 / 3)
    assert float(summary["all_loci_recall"]) == pytest.approx(2 / 3)
    assert float(summary["unmasked_loci_recall"]) == 0.5
    assert float(summary["unmasked_training_family_loci_recall"]) == 1.0
    assert float(summary["unmasked_heldout_family_loci_recall"]) == 0.0
    # unambiguous candidates ranked: c1 (positive, 50), c3 (negative, 20), c2 and c4 tied without a hit
    assert float(summary["ap_unambiguous_candidates"]) == pytest.approx(0.5 * 1 + 0.5 * 0.5)
    assert float(summary["precision_unambiguous_calls"]) == 0.5
    scores = {r["record_id"]: (float(r["score"]), r["called"]) for r in csv.DictReader(open(tmp_path / "predictions.csv"))}
    assert scores == {"c1": (50.0, "1"), "c2": (cac.NO_HIT_SCORE, "0"), "c3": (20.0, "1"), "c4": (cac.NO_HIT_SCORE, "0"), "c5": (30.0, "1")}
