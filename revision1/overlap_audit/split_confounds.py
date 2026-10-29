#!/usr/bin/env python3
"""Quantify two evaluation confounds in pre-miRBench.

1. Negative swap: AP of one test's positives scored against another test's negatives
   (species-matched, same 1:10 ratio). Isolates how much the negative set, which was
   allocated to partitions in mining-score order, contributes to between-test AP differences.
2. Cluster bootstrap: AP CIs when overlapping test windows (either strand) are resampled
   together, instead of as independent records.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

SPLIT_NAME = {
    "test_known_species_known_family": "Test 1",
    "test_known_species_heldout_family": "Test 2",
    "test_heldout_species_known_family": "Test 3",
    "test_heldout_species_heldout_family": "Test 4",
}
TOOL_NAME = {
    "deepmir": "DeepMir", "deepmirgene": "deepMiRGene", "dnnpremir": "dnnPreMiR", "mirdnn": "mirDNN",
    "mire2e": "miRe2e", "mustard": "MuStARD", "premirbench": "pre-miRBench model",
}
SWAPS = [
    ("test_known_species_heldout_family", "test_known_species_known_family"),
    ("test_known_species_known_family", "test_known_species_heldout_family"),
    ("test_heldout_species_heldout_family", "test_heldout_species_known_family"),
    ("test_heldout_species_known_family", "test_heldout_species_heldout_family"),
]


def paired_species_matched_sample(positives, own_negatives, other_negatives, ratio, rng):
    """Per species, k = min(positives, own//ratio, other//ratio): the same k positives, with ratio*k negatives from each pool."""
    pos_index, own_index, other_index = [], [], []
    for species, count in positives["species"].value_counts().items():
        own_pool = own_negatives.index[own_negatives["species"] == species].to_numpy()
        other_pool = other_negatives.index[other_negatives["species"] == species].to_numpy()
        k = min(count, len(own_pool) // ratio, len(other_pool) // ratio)
        if k == 0:
            continue
        pos_index.append(rng.choice(positives.index[positives["species"] == species].to_numpy(), k, replace=False))
        own_index.append(rng.choice(own_pool, ratio * k, replace=False))
        other_index.append(rng.choice(other_pool, ratio * k, replace=False))
    return np.concatenate(pos_index), np.concatenate(own_index), np.concatenate(other_index)


def negative_swap(merged, draws, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for tool, tool_rows in merged.groupby("tool"):
        for pos_split, neg_split in SWAPS:
            positives = tool_rows[(tool_rows["split"] == pos_split) & (tool_rows["label"] == 1)]
            own_negatives = tool_rows[(tool_rows["split"] == pos_split) & (tool_rows["label"] == 0)]
            other_negatives = tool_rows[(tool_rows["split"] == neg_split) & (tool_rows["label"] == 0)]
            own, swapped = [], []
            for _ in range(draws):
                p, n_own, n_other = paired_species_matched_sample(positives, own_negatives, other_negatives, 10, rng)
                for index, store in ((n_own, own), (n_other, swapped)):
                    y = np.r_[np.ones(len(p)), np.zeros(len(index))]
                    s = np.r_[tool_rows.loc[p, "score"].to_numpy(), tool_rows.loc[index, "score"].to_numpy()]
                    store.append(average_precision_score(y, s))
            rows.append(
                {
                    "tool": tool,
                    "positives_from": SPLIT_NAME[pos_split],
                    "negatives_from": SPLIT_NAME[neg_split],
                    "positives_per_draw": len(p),
                    "ap_own_negatives_species_matched": np.mean(own),
                    "ap_swapped_negatives_species_matched": np.mean(swapped),
                    "difference_swapped_minus_own": np.mean(swapped) - np.mean(own),
                }
            )
    return pd.DataFrame(rows)


def overlap_clusters(dataset):
    """Connected components of test records whose windows overlap (either strand), within each test."""
    cluster = pd.Series(index=dataset.index, dtype=object)
    for (split, chrom), group in dataset.groupby(["split", "chrom"]):
        group = group.sort_values("start")
        current_end, current_id = -1, None
        for index, start, end in group[["start", "end"]].itertuples():
            if start >= current_end:
                current_id = f"{split}|{chrom}|{start}"
            current_end = max(current_end, end) if start < current_end else end
            cluster[index] = current_id
    return cluster


def cluster_bootstrap(merged, dataset, resamples, seed):
    tests = dataset[dataset["split"].isin(SPLIT_NAME)].copy()
    tests["cluster"] = overlap_clusters(tests)
    cluster_of = tests.set_index("record_id")["cluster"]
    rng = np.random.default_rng(seed)
    rows = []
    for split, name in SPLIT_NAME.items():
        part = merged[merged["split"] == split]
        wide = part.pivot(index="record_id", columns="tool", values="score")
        labels = part.drop_duplicates("record_id").set_index("record_id")["label"].reindex(wide.index).to_numpy()
        clusters = cluster_of.reindex(wide.index).to_numpy()
        # Stratify by label as in the manuscript: resample positive clusters and negative clusters separately.
        strata = {}
        for label in (0, 1):
            ids = pd.Series(np.flatnonzero(labels == label)).groupby(clusters[labels == label]).apply(list)
            strata[label] = list(ids)
        n_records = len(labels)
        n_clusters = sum(len(v) for v in strata.values())
        boot = {tool: [] for tool in wide.columns}
        diff = []
        scores = {tool: wide[tool].to_numpy() for tool in wide.columns}
        for _ in range(resamples):
            index = []
            for label in (0, 1):
                groups = strata[label]
                pick = rng.integers(0, len(groups), len(groups))
                index.extend(i for g in pick for i in groups[g])
            index = np.array(index)
            values = {tool: average_precision_score(labels[index], scores[tool][index]) for tool in wide.columns}
            for tool in wide.columns:
                boot[tool].append(values[tool])
            diff.append(values["pre-miRBench model"] - values["dnnPreMiR"])
        for tool in wide.columns:
            rows.append(
                {
                    "test": name, "quantity": f"AP {tool}", "records": n_records, "clusters": n_clusters,
                    "point": average_precision_score(labels, scores[tool]),
                    "ci_2.5": np.percentile(boot[tool], 2.5), "ci_97.5": np.percentile(boot[tool], 97.5),
                }
            )
        rows.append(
            {
                "test": name, "quantity": "AP difference pre-miRBench minus dnnPreMiR", "records": n_records, "clusters": n_clusters,
                "point": average_precision_score(labels, scores["pre-miRBench model"]) - average_precision_score(labels, scores["dnnPreMiR"]),
                "ci_2.5": np.percentile(diff, 2.5), "ci_97.5": np.percentile(diff, 97.5),
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--published-predictions", required=True)
    parser.add_argument("--premirbench-predictions", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = pd.read_csv(args.dataset, keep_default_na=False, usecols=["record_id", "species", "split", "chrom", "start", "end", "label"])
    dataset[["start", "end", "label"]] = dataset[["start", "end", "label"]].astype(int)
    predictions = pd.concat([pd.read_csv(args.published_predictions), pd.read_csv(args.premirbench_predictions)], ignore_index=True)
    predictions["tool"] = predictions["tool"].map(TOOL_NAME)
    merged = predictions.drop(columns=["species", "label"]).merge(dataset, on=["record_id", "split"], validate="many_to_one")
    assert len(merged) == len(predictions)

    negative_swap(merged, args.draws, args.seed).to_csv(f"{args.output_prefix}negative_swap_ap.tsv", sep="\t", index=False, float_format="%.4f")
    cluster_bootstrap(merged, dataset, args.resamples, args.seed).to_csv(
        f"{args.output_prefix}cluster_bootstrap_ap.tsv", sep="\t", index=False, float_format="%.4f"
    )


if __name__ == "__main__":
    main()
