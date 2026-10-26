import sys
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comparators"))

import mirmachine_retrained as mr  # noqa: E402


def test_trusted_cutoff_follows_the_mirmachine_formula():
    # (bitscore, is true positive), in descending bitscore order as cmsearch reports them
    hits = [(80.0, True), (70.0, True), (60.0, False), (50.0, True), (40.0, False), (30.0, False)]
    # sqrt(TP/(TP+FN+1) * TP/(TP+FP)) at each candidate cutoff
    expected = {
        80.0: sqrt(1 / 4 * 1 / 1), 70.0: sqrt(2 / 4 * 2 / 2), 60.0: sqrt(2 / 4 * 2 / 3),
        50.0: sqrt(3 / 4 * 3 / 4), 40.0: sqrt(3 / 4 * 3 / 5), 30.0: sqrt(3 / 4 * 3 / 6),
    }
    assert mr.trusted_cutoff(hits) == max(expected, key=expected.get) == 50.0


def test_precursor_interval_in_window_coordinates():
    plus = {"start": "1000", "end": "1200", "strand": "+", "target_start": "1070", "target_end": "1130"}
    minus = {"start": "1000", "end": "1200", "strand": "-", "target_start": "1050", "target_end": "1110"}
    assert mr.precursor_interval(plus) == (70, 130)
    assert mr.precursor_interval(minus) == (90, 150)


def test_genome_hit_labels_use_training_information_only():
    training = mr.interval_index([("s__chr1", 1000, 1060, ("+", "Mir-1"))])
    evaluation = mr.interval_index([("s__chr1", 5000, 5200, "s_pos_000002")])  # a validation or test window
    annotated = mr.interval_index([("s__chr1", 1000, 1060, "S-Mir-1_pre"), ("s__chr1", 9000, 9060, "S-Mir-2_pre")])

    def label(start, end, strand="+", family="Mir-1"):
        hit = {"target": "s__chr1", "start": start, "end": end, "strand": strand, "family": family}
        return mr.label_genome_hit(hit, training, evaluation, annotated, margin=100)

    assert label(990, 1070) == "TP"  # training precursor, same family and strand
    assert label(990, 1070, strand="-") == "FP"  # antisense to the training precursor
    assert label(990, 1070, family="Mir-2") == "FP"  # training precursor of another family
    assert label(5050, 5120) == "excluded_evaluation_window"  # inside a validation or test window
    assert label(4850, 4910, strand="-") == "excluded_evaluation_window"  # within the margin, other strand
    assert label(4800, 4890) == "FP"  # 110 nt before the window: outside the margin
    assert label(9000, 9060, family="Mir-2") == "excluded_other_annotation"  # annotated, not in training
    assert label(20000, 20070) == "FP"


def test_evaluation_windows_take_precedence_over_training_precursors():
    training = mr.interval_index([("s__chr1", 1000, 1060, ("+", "Mir-1"))])
    evaluation = mr.interval_index([("s__chr1", 1100, 1300, "s_neg_000001")])
    hit = {"target": "s__chr1", "start": 1000, "end": 1060, "strand": "+", "family": "Mir-1"}
    assert mr.label_genome_hit(hit, training, evaluation, mr.interval_index([]), margin=100) == "excluded_evaluation_window"
