#!/usr/bin/env python3
"""Recompute the numbers reported in the submitted pre-miRBench manuscript from released files."""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

SPLIT_NAME = {
    "train": "Train",
    "valid": "Validation",
    "test_known_species_known_family": "Test 1",
    "test_known_species_heldout_family": "Test 2",
    "test_heldout_species_known_family": "Test 3",
    "test_heldout_species_heldout_family": "Test 4",
}
TOOL_NAME = {
    "deepmir": "DeepMir", "deepmirgene": "deepMiRGene", "dnnpremir": "dnnPreMiR", "mirdnn": "mirDNN",
    "mire2e": "miRe2e", "mustard": "MuStARD", "premirbench": "pre-miRBench model",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-summary", required=True)
    parser.add_argument("--published-predictions", required=True)
    parser.add_argument("--premirbench-predictions", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.dataset, keep_default_na=False)
    df["label"] = df["label"].astype(int)

    print("== Table 1")
    pos = df[df["label"] == 1]
    for split, name in SPLIT_NAME.items():
        s = df[df["split"] == split]
        p = s[s["label"] == 1]
        print(f"{name:11s} species={s['species'].nunique():3d} families={p['family_id'].nunique():4d} pos={len(p):5d} neg={(s['label']==0).sum():6d} total={len(s):6d}")
    print(f"Total       species={df['species'].nunique()} families={pos['family_id'].nunique()} pos={len(pos)} neg={(df['label']==0).sum()} total={len(df)}")

    print("== Build counts (species summary sums)")
    summary = pd.read_csv(args.split_summary)
    for column in ["bed_rows", "bed_pre_entries", "positives_after_filters", "positive_boundary_filtered", "positive_repeat_filtered",
                   "negative_windows_folded", "negative_hairpin_like_kept", "negative_overlap_skipped", "negative_repeat_or_n_skipped", "excluded_pos"]:
        print(f"{column}: {pd.to_numeric(summary[column], errors='coerce').sum():,.0f}")

    predictions = pd.concat([pd.read_csv(args.published_predictions), pd.read_csv(args.premirbench_predictions)])
    predictions["tool"] = predictions["tool"].map(TOOL_NAME)
    print(f"== predictions rows={len(predictions)} missing_scores={predictions['score'].isna().sum()}")

    print("== Table 2 AP and mean AP")
    ap = predictions.groupby(["tool", "split"]).apply(lambda g: average_precision_score(g["label"], g["score"])).unstack()
    ap = ap[list(SPLIT_NAME)[2:]]
    ap.columns = [SPLIT_NAME[c] for c in ap.columns]
    ap["mean"] = ap.mean(axis=1)
    print(ap.round(4).to_string())

    print("== Table 3 (Test 4, threshold 0.5)")
    t4 = predictions[predictions["split"] == "test_heldout_species_heldout_family"]
    for tool, g in t4.groupby("tool"):
        y, pred = g["label"].to_numpy(), (g["score"].to_numpy() >= 0.5).astype(int)
        tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
        fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
        stored = int((g["predicted_label"] != pred).sum())
        prec, rec = tp / (tp + fp) if tp + fp else np.nan, tp / (tp + fn)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        print(f"{tool:20s} TP={tp} FP={fp} FN={fn} TN={tn} acc={(tp+tn)/len(y):.4f} prec={prec:.4f} rec={rec:.4f} spec={tn/(tn+fp):.4f} "
              f"F1={2*prec*rec/(prec+rec):.4f} MCC={mcc:.4f} stored_label_mismatch={stored} n_score_eq_0.5={(g['score']==0.5).sum()}")

    print("== Test 4 AP by species")
    for tool, g in t4.groupby("tool"):
        by = {sp: average_precision_score(h["label"], h["score"]) for sp, h in g.groupby("species")}
        print(f"{tool:20s} " + " ".join(f"{sp}={v:.3f} (pos={int(g[g.species==sp].label.sum())})" for sp, v in by.items()))

    print("== Test 4: highest recall with precision >= 0.95 (and counts)")
    for tool, g in t4.groupby("tool"):
        precision, recall, thresholds = precision_recall_curve(g["label"], g["score"])
        ok = np.flatnonzero(precision[:-1] >= 0.95)
        i = ok[np.argmax(recall[ok])]
        thr = thresholds[i]
        sel = g["score"] >= thr
        tp, fp = int((sel & (g["label"] == 1)).sum()), int((sel & (g["label"] == 0)).sum())
        print(f"{tool:20s} threshold={thr:.4g} TP={tp} FP={fp} precision={precision[i]:.3f} recall={recall[i]:.3f}")

    print("== score ties (fraction of records sharing an exact score with another record in the same test)")
    for (tool, split), g in predictions.groupby(["tool", "split"]):
        dup = g["score"].duplicated(keep=False).mean()
        if dup > 0.05:
            print(f"{tool:20s} {SPLIT_NAME[split]} tied_fraction={dup:.3f} distinct_scores={g['score'].nunique()} max_score_count={(g['score']==g['score'].max()).sum()}")


if __name__ == "__main__":
    main()
