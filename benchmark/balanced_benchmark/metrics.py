#!/usr/bin/env python3
import math

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


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


def compute_auc_metrics(rows):
    """Compute ROC AUC and PR AUC from rows that contain a continuous score.

    Returns None for both if all scores are identical (no discrimination power)
    or if scores are missing.

    Also returns the (fpr, tpr) and (precision_curve, recall_curve) arrays
    needed to plot the curves.
    """
    scores = [row.get("score") for row in rows]
    labels = [int(row["ground_truth_class"]) for row in rows]

    # Check we have real scores (not None) and that both classes are present
    if any(s is None for s in scores):
        return {
            "roc_auc": None, "pr_auc": None,
            "fpr": None, "tpr": None,
            "precision_curve": None, "recall_curve": None,
        }

    scores = [float(s) for s in scores]

    if len(set(labels)) < 2:
        return {
            "roc_auc": None, "pr_auc": None,
            "fpr": None, "tpr": None,
            "precision_curve": None, "recall_curve": None,
        }

    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    prec_curve, rec_curve, _ = precision_recall_curve(labels, scores)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": prec_curve.tolist(),
        "recall_curve": rec_curve.tolist(),
    }
