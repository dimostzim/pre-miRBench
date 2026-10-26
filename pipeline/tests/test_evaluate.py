import random
import sys
from pathlib import Path

import pytest
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate as ev  # noqa: E402


def brute_force_best_f1(labels, scores):
    best = {"best_f1": None, "best_threshold": None}
    for threshold in sorted(set(scores), reverse=True):
        f1 = ev.confusion_metrics(labels, scores, threshold)["f1"]
        if f1 is not None and (best["best_f1"] is None or f1 > best["best_f1"]):
            best = {"best_f1": f1, "best_threshold": threshold}
    return best


def test_best_f1_matches_threshold_scan_including_ties_and_no_positives():
    rng = random.Random(3)
    for trial in range(200):
        n = rng.randint(1, 60)
        labels = [int(rng.random() < 0.3) for _ in range(n)] if trial % 10 else [0] * n
        scores = [round(rng.random(), 1) for _ in range(n)]  # coarse scores create ties
        assert ev.best_f1(labels, scores) == brute_force_best_f1(labels, scores)


def test_average_precision_groups_tied_scores_independently_of_record_order():
    rng = random.Random(5)
    for _ in range(200):
        n = rng.randint(2, 60)
        labels = [int(rng.random() < 0.3) for _ in range(n)]
        labels[0] = 1
        scores = [round(rng.random(), 1) for _ in range(n)]  # ties across both classes
        expected = average_precision_score(labels, scores)
        assert ev.average_precision(labels, scores) == pytest.approx(expected)
        assert ev.average_precision(labels[::-1], scores[::-1]) == pytest.approx(expected)
