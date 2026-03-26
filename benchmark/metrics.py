#!/usr/bin/env python3
import math


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def matthews_corrcoef(tp, fp, tn, fn):
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if not denominator:
        return 0.0
    return ((tp * tn) - (fp * fn)) / denominator


def compute_binary_metrics(rows):
    tp = fp = tn = fn = 0

    for row in rows:
        pred = int(row["predicted_class"])
        gt = int(row["ground_truth_class"])

        if pred == 1 and gt == 1:
            tp += 1
        elif pred == 1 and gt == 0:
            fp += 1
        elif pred == 0 and gt == 0:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, total)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    mcc = matthews_corrcoef(tp, fp, tn, fn)

    return {
        "n": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "mcc": mcc,
    }
