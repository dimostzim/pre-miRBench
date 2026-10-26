import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_dataset as bd  # noqa: E402


def test_stratified_labels_give_every_partition_the_same_rank_distribution():
    quotas = {"train": 700, "valid": 100, "test_known_species_known_family": 100, "test_known_species_heldout_family": 100}
    labels = bd.stratified_split_labels(quotas, seed=1)
    assert Counter(labels) == Counter(quotas)
    # Mean rank of each partition should sit near the middle of the ranked list (no partition gets the top).
    mean_rank = {split: sum(i for i, label in enumerate(labels) if label == split) / quotas[split] for split in quotas}
    assert all(abs(value - len(labels) / 2) < 0.02 * len(labels) for value in mean_rank.values())


def test_leakage_key_matches_reverse_complement():
    forward = {"canonical_100nt_sequence": "AACGU"}
    reverse = {"canonical_100nt_sequence": "ACGUU"}
    assert bd.leakage_key(forward) == bd.leakage_key(reverse)


def positive(record, split, chrom, start, family="Mir-1", species="hsa"):
    return {
        "record_id": record, "split": split, "chrom": chrom, "start": str(start), "end": str(start + 200),
        "species": species, "family_id": family, "mirna_id": record, "window_id": record,
    }


def test_overlapping_positive_windows_keep_the_higher_priority_partition():
    rows = [
        positive("train_a", "train", "chr1", 1000),
        positive("test_a", "test_known_species_heldout_family", "chr1", 1150),  # overlaps train_a
        positive("train_b", "train", "chr1", 1300),  # overlaps test_a only, same split as train_a
        positive("valid_c", "valid", "chr2", 5000),
    ]
    kept, excluded = bd.remove_cross_split_positive_overlaps(rows)
    assert {row["record_id"] for row in kept} == {"test_a", "valid_c"}
    assert {row["record_id"] for row in excluded} == {"train_a", "train_b"}
    assert all(row["split_reason"] == "overlapping_window_excluded" for row in excluded)


def test_window_overlap_counts_detect_any_strand_overlap():
    rows = [
        {**positive("p", "train", "chr1", 0), "label": "1"},
        {**positive("n1", "train", "chr1", 150), "label": "0"},
        {**positive("n2", "valid", "chr1", 300), "label": "0"},
    ]
    counts = bd.window_overlap_counts(rows)
    assert counts["negative_positive"] == 1 and counts["negative_negative"] == 1 and counts["cross_split"] == 1


def test_heldout_species_mine_with_a_typical_sized_sample_of_pooled_positives():
    train = [
        {"species": species, "window_id": f"{species}{i}"}
        for species, count in (("aaa", 10), ("bbb", 30), ("ccc", 50))
        for i in range(count)
    ]
    sets = bd.mining_positive_sets(train, ["aaa", "bbb", "ccc", "zzz"], seed=42)
    assert len(sets["aaa"]) == 10 and all(row["species"] == "aaa" for row in sets["aaa"])
    assert len(sets["zzz"]) == 30  # median per-species training size
    assert sets["zzz"] == bd.mining_positive_sets(train, ["aaa", "bbb", "ccc", "zzz"], seed=42)["zzz"]


def test_positive_split_assignment_does_not_depend_on_input_order():
    import argparse
    import random

    rows = []
    for species in ("aaa", "bbb", "ccc", "hhh"):
        for family in range(12):
            for copy in range(3):
                rows.append({
                    "species": species, "chrom": f"{species}__c", "start": str(1000 * (family * 3 + copy)), "end": str(1000 * (family * 3 + copy) + 200),
                    "strand": "+", "mirna_id": f"Mir-{family}-P{copy}", "family_id": f"Mir-{family}",
                    "precursor_sequence": f"{species}{family}{copy}", "canonical_100nt_sequence": f"{species}{family}{copy}",
                    "window_id": f"{species}|{family}|{copy}", "split": "", "split_reason": "",
                })
    args = argparse.Namespace(heldout_species="hhh", valid_frac=0.1, valid_heldout_family_frac=0.0,
                              test_known_species_known_family_frac=0.1, test_known_species_heldout_family_frac=0.1, seed=42)
    first, _ = bd.assign_positive_splits([dict(row) for row in rows], args)
    shuffled = [dict(row) for row in rows]
    random.Random(7).shuffle(shuffled)
    second, _ = bd.assign_positive_splits(shuffled, args)
    assert {row["window_id"]: row["split"] for row in first} == {row["window_id"]: row["split"] for row in second}


def test_tool_crops_centre_on_the_hairpin_when_the_record_has_one():
    # Hairpin 100-160 and annotated precursor 50-110 within a 200-nt window starting at 1000.
    row = {"start": "1000", "hairpin_start": "1100", "hairpin_end": "1160", "target_start": "1050", "target_end": "1110"}
    assert bd.tool_crop_bounds(row, 200, 100) == (80, 180)
    assert bd.tool_crop_bounds({**row, "hairpin_start": "", "hairpin_end": ""}, 200, 100) == (30, 130)
