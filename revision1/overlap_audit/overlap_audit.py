#!/usr/bin/env python3
"""Audit genomic, reverse-complement and selection overlap between pre-miRBench partitions.

Inputs are the canonical dataset.csv, the combined genome index and the released
record-level predictions. Outputs are TSV tables in --output-dir.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

SPLITS = [
    "train",
    "valid",
    "test_known_species_known_family",
    "test_known_species_heldout_family",
    "test_heldout_species_known_family",
    "test_heldout_species_heldout_family",
]
SPLIT_NAME = {
    "train": "Train",
    "valid": "Validation",
    "test_known_species_known_family": "Test 1",
    "test_known_species_heldout_family": "Test 2",
    "test_heldout_species_known_family": "Test 3",
    "test_heldout_species_heldout_family": "Test 4",
}
TESTS = SPLITS[2:]
TOOL_NAME = {
    "deepmir": "DeepMir",
    "deepmirgene": "deepMiRGene",
    "dnnpremir": "dnnPreMiR",
    "mirdnn": "mirDNN",
    "mire2e": "miRe2e",
    "mustard": "MuStARD",
    "premirbench": "pre-miRBench model",
}
COMPLEMENT = str.maketrans("ACGUN", "UGCAN")


def reverse_complement(sequence):
    return sequence.upper().translate(COMPLEMENT)[::-1]


def load_dataset(path):
    df = pd.read_csv(path, dtype={"label": int, "hard_round": "Int64"}, keep_default_na=False, na_values={"hard_round": [""]})
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["sequence"] = df["sequence"].str.upper()
    df["label_name"] = np.where(df["label"] == 1, "pos", "neg")
    df["split_name"] = df["split"].map(SPLIT_NAME)
    return df.reset_index(drop=True)


def overlapping_pairs(df):
    """Return every pair of records whose genomic windows share at least 1 nt (either strand)."""
    rows = []
    for _chrom, group in df.groupby("chrom", sort=False):
        group = group.sort_values("start")
        index = group.index.to_numpy()
        start = group["start"].to_numpy()
        end = group["end"].to_numpy()
        for i in range(len(group)):
            j = i + 1
            while j < len(group) and start[j] < end[i]:
                rows.append((index[i], index[j], min(end[i], end[j]) - start[j]))
                j += 1
    pairs = pd.DataFrame(rows, columns=["a", "b", "overlap_nt"])
    for side in ("a", "b"):
        for column in ("split", "label_name", "strand", "species", "start", "end"):
            pairs[f"{column}_{side}"] = df.loc[pairs[side], column].to_numpy()
    pairs["same_strand"] = pairs["strand_a"] == pairs["strand_b"]
    pairs["identical_coordinates"] = (pairs["start_a"] == pairs["start_b"]) & (pairs["end_a"] == pairs["end_b"])
    return pairs


def pair_category(split_a, split_b):
    if split_a == split_b:
        return "within " + SPLIT_NAME[split_a]
    ordered = sorted((split_a, split_b), key=SPLITS.index)
    return f"{SPLIT_NAME[ordered[0]]} vs {SPLIT_NAME[ordered[1]]}"


def summarize_pairs(pairs):
    pairs = pairs.copy()
    pairs["category"] = [pair_category(a, b) for a, b in zip(pairs["split_a"], pairs["split_b"])]
    pairs["labels"] = ["-".join(sorted((a, b))) for a, b in zip(pairs["label_name_a"], pairs["label_name_b"])]
    summary = (
        pairs.groupby(["category", "labels"])
        .agg(
            pairs=("overlap_nt", "size"),
            same_strand=("same_strand", "sum"),
            exact_reverse_complement=("identical_coordinates", lambda s: int((s & ~pairs.loc[s.index, "same_strand"]).sum())),
            overlap_ge_100nt=("overlap_nt", lambda s: int((s >= 100).sum())),
            overlap_ge_150nt=("overlap_nt", lambda s: int((s >= 150).sum())),
            median_overlap_nt=("overlap_nt", "median"),
        )
        .reset_index()
    )
    return pairs, summary


def per_record_max_overlap(df, pairs, reference_splits):
    """For each record, the largest genomic overlap with any record in reference_splits."""
    best = np.zeros(len(df), dtype=int)
    best_rc = np.zeros(len(df), dtype=bool)
    reference = set(reference_splits)
    for a, b, overlap, split_a, split_b, identical, same in pairs[
        ["a", "b", "overlap_nt", "split_a", "split_b", "identical_coordinates", "same_strand"]
    ].itertuples(index=False):
        if split_b in reference and split_a != split_b:
            best[a] = max(best[a], overlap)
            best_rc[a] |= identical and not same
        if split_a in reference and split_a != split_b:
            best[b] = max(best[b], overlap)
            best_rc[b] |= identical and not same
    return best, best_rc


def exact_sequence_matches(df):
    """Exact full-window and 100-nt matches, forward or reverse complement, between different records."""
    rows = []
    for field in ("sequence", "canonical_100nt_sequence"):
        forward = defaultdict(list)
        for index, sequence in df[field].items():
            forward[sequence.upper()].append(index)
        for index, sequence in df[field].items():
            for other in forward.get(reverse_complement(sequence), []):
                if other > index:
                    rows.append((field, "reverse_complement", index, other))
        for indices in forward.values():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    rows.append((field, "forward", indices[i], indices[j]))
    matches = pd.DataFrame(rows, columns=["field", "orientation", "a", "b"])
    for side in ("a", "b"):
        for column in ("record_id", "split", "label_name"):
            matches[f"{column}_{side}"] = df.loc[matches[side], column].to_numpy()
    matches["category"] = [pair_category(a, b) for a, b in zip(matches["split_a"], matches["split_b"])]
    return matches


def negative_difficulty(df):
    negatives = df[df["label"] == 0].copy()
    negatives["is_mined_hard"] = negatives["hard_round"].notna()
    by_split = (
        negatives.groupby("split", sort=False)
        .agg(
            negatives=("score", "size"),
            mined_hard_fraction=("is_mined_hard", "mean"),
            median_mining_score=("score", "median"),
            mean_mining_score=("score", "mean"),
        )
        .reindex(SPLITS)
        .reset_index()
    )
    # Within-species percentile of each negative's mining score among the species' selected negatives,
    # so species with different score scales are comparable.
    negatives["species_score_percentile"] = negatives.groupby("species")["score"].rank(pct=True)
    percentile = negatives.groupby("split")["species_score_percentile"].median().rename("median_within_species_percentile")
    by_split = by_split.merge(percentile, left_on="split", right_index=True)
    rounds = negatives.groupby(["split", "hard_round"], dropna=False).size().unstack(fill_value=0)
    rounds.columns = [f"hard_round_{c}" if pd.notna(c) else "not_mined_hard" for c in rounds.columns]
    by_split = by_split.merge(rounds, left_on="split", right_index=True)
    by_split.insert(1, "split_name", by_split["split"].map(SPLIT_NAME))
    return by_split


def chromosome_position(df, fai_path):
    lengths = pd.read_csv(fai_path, sep="\t", header=None, usecols=[0, 1], names=["chrom", "length"])
    lengths = dict(zip(lengths["chrom"], lengths["length"]))
    df = df.copy()
    df["chrom_length"] = df["chrom"].map(lengths)
    df["relative_position"] = df["start"] / df["chrom_length"]
    summary = (
        df.groupby("label_name")
        .agg(
            records=("start", "size"),
            median_start_nt=("start", "median"),
            fraction_start_within_1mb=("start", lambda s: float((s < 1_000_000).mean())),
            median_relative_position=("relative_position", "median"),
            fraction_in_first_1pct_of_chrom=("relative_position", lambda s: float((s < 0.01).mean())),
        )
        .reset_index()
    )
    per_species = (
        df[df["label"] == 0]
        .groupby("species")
        .agg(
            negatives=("start", "size"),
            negative_chromosomes=("chrom", "nunique"),
            max_negative_start_nt=("start", "max"),
            median_negative_start_nt=("start", "median"),
        )
        .reset_index()
    )
    positives_per_species = (
        df[df["label"] == 1].groupby("species").agg(median_positive_start_nt=("start", "median")).reset_index()
    )
    return summary, per_species.merge(positives_per_species, on="species")


def family_overlap(df):
    positives = df[df["label"] == 1]
    families = {split: set(positives.loc[positives["split"] == split, "family_id"]) for split in SPLITS}
    rows = []
    for a in SPLITS:
        for b in SPLITS:
            rows.append(
                {
                    "split_a": SPLIT_NAME[a],
                    "split_b": SPLIT_NAME[b],
                    "families_a": len(families[a]),
                    "families_b": len(families[b]),
                    "shared_families": len(families[a] & families[b]),
                }
            )
    return pd.DataFrame(rows)


def load_predictions(published, premirbench):
    predictions = pd.concat([pd.read_csv(published), pd.read_csv(premirbench)], ignore_index=True)
    predictions["tool"] = predictions["tool"].map(TOOL_NAME)
    return predictions


def ap_with_exclusions(df, predictions, flags, seed, draws):
    """AP on each test after removing flagged test records, with a random-removal control of equal size and label mix."""
    merged = predictions.merge(df[["record_id", *flags]], on="record_id", how="left", validate="many_to_one")
    rng = np.random.default_rng(seed)
    rows = []
    for (tool, split), group in merged.groupby(["tool", "split"]):
        y = group["label"].to_numpy()
        s = group["score"].to_numpy()
        full_ap = average_precision_score(y, s)
        for flag in flags:
            mask = group[flag].to_numpy().astype(bool)
            kept = ~mask
            removed_pos = int((mask & (y == 1)).sum())
            removed_neg = int((mask & (y == 0)).sum())
            filtered_ap = average_precision_score(y[kept], s[kept]) if kept.any() and y[kept].any() else np.nan
            # Control: remove the same number of positives and negatives, chosen at random from unflagged records.
            control = []
            pos_pool = np.flatnonzero(~mask & (y == 1))
            neg_pool = np.flatnonzero(~mask & (y == 0))
            if removed_pos <= len(pos_pool) and removed_neg <= len(neg_pool) and (removed_pos or removed_neg):
                for _ in range(draws):
                    drop = np.concatenate(
                        [rng.choice(pos_pool, removed_pos, replace=False), rng.choice(neg_pool, removed_neg, replace=False)]
                    )
                    keep = np.ones(len(y), dtype=bool)
                    keep[drop] = False
                    control.append(average_precision_score(y[keep], s[keep]))
            rows.append(
                {
                    "tool": tool,
                    "split": split,
                    "test": SPLIT_NAME[split],
                    "exclusion": flag,
                    "removed_positives": removed_pos,
                    "removed_negatives": removed_neg,
                    "ap_full": full_ap,
                    "ap_after_exclusion": filtered_ap,
                    "ap_random_removal_control_mean": float(np.mean(control)) if control else np.nan,
                    "ap_random_removal_control_2.5": float(np.percentile(control, 2.5)) if control else np.nan,
                    "ap_random_removal_control_97.5": float(np.percentile(control, 97.5)) if control else np.nan,
                    "mean_score_flagged_negatives": float(s[mask & (y == 0)].mean()) if removed_neg else np.nan,
                    "mean_score_unflagged_negatives": float(s[~mask & (y == 0)].mean()),
                }
            )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fai", required=True)
    parser.add_argument("--published-predictions", default=None, help="Omit to audit the dataset only.")
    parser.add_argument("--premirbench-predictions", default=None)
    parser.add_argument("--expected-records", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    assert df["record_id"].is_unique
    if args.expected_records is not None:
        assert len(df) == args.expected_records, len(df)

    pairs, pair_summary = summarize_pairs(overlapping_pairs(df))
    pair_summary.to_csv(out / "genomic_overlap_pairs_summary.tsv", sep="\t", index=False)
    pairs.drop(columns=["a", "b"]).assign(
        record_a=df.loc[pairs["a"], "record_id"].to_numpy(), record_b=df.loc[pairs["b"], "record_id"].to_numpy()
    ).to_csv(out / "genomic_overlap_pairs.tsv.gz", sep="\t", index=False)

    max_train, rc_train = per_record_max_overlap(df, pairs, ["train"])
    max_train_valid, rc_train_valid = per_record_max_overlap(df, pairs, ["train", "valid"])
    df["max_overlap_with_train_nt"] = max_train
    df["exact_rc_of_train_record"] = rc_train
    df["max_overlap_with_train_or_valid_nt"] = max_train_valid
    df["exact_rc_of_train_or_valid_record"] = rc_train_valid

    rows = []
    for split in SPLITS[1:]:
        for label in ("pos", "neg"):
            subset = df[(df["split"] == split) & (df["label_name"] == label)]
            reference = subset["max_overlap_with_train_nt"]
            rows.append(
                {
                    "split": SPLIT_NAME[split],
                    "label": label,
                    "records": len(subset),
                    "overlap_train_any_nt": int((reference > 0).sum()),
                    "overlap_train_ge_50nt": int((reference >= 50).sum()),
                    "overlap_train_ge_100nt": int((reference >= 100).sum()),
                    "overlap_train_ge_150nt": int((reference >= 150).sum()),
                    "exact_rc_of_train_record": int(subset["exact_rc_of_train_record"].sum()),
                    "overlap_train_or_valid_any_nt": int((subset["max_overlap_with_train_or_valid_nt"] > 0).sum()),
                }
            )
    per_record = pd.DataFrame(rows)
    per_record["fraction_overlap_train_any_nt"] = per_record["overlap_train_any_nt"] / per_record["records"]
    per_record.to_csv(out / "records_overlapping_train.tsv", sep="\t", index=False)

    matches = exact_sequence_matches(df)
    matches.groupby(["field", "orientation", "category"]).size().rename("pairs").reset_index().to_csv(
        out / "exact_sequence_matches_summary.tsv", sep="\t", index=False
    )
    matches.to_csv(out / "exact_sequence_matches.tsv", sep="\t", index=False)

    negative_difficulty(df).to_csv(out / "negative_difficulty_by_split.tsv", sep="\t", index=False)

    position_summary, position_species = chromosome_position(df, args.fai)
    position_summary.to_csv(out / "chromosome_position_summary.tsv", sep="\t", index=False)
    position_species.to_csv(out / "chromosome_position_by_species.tsv", sep="\t", index=False)

    family_overlap(df).to_csv(out / "family_overlap_between_splits.tsv", sep="\t", index=False)

    df["flag_overlap_train_any"] = df["max_overlap_with_train_nt"] > 0
    df["flag_overlap_train_or_valid_any"] = df["max_overlap_with_train_or_valid_nt"] > 0
    df["flag_overlap_train_ge_100nt"] = df["max_overlap_with_train_nt"] >= 100
    df["flag_exact_rc_train"] = df["exact_rc_of_train_record"]
    flags = ["flag_overlap_train_any", "flag_overlap_train_or_valid_any", "flag_overlap_train_ge_100nt", "flag_exact_rc_train"]
    df[["record_id", "split", "label", "species", *flags, "max_overlap_with_train_nt", "max_overlap_with_train_or_valid_nt"]].to_csv(
        out / "record_overlap_flags.tsv.gz", sep="\t", index=False
    )

    if not args.published_predictions:
        print(f"wrote {out} (dataset audit only)")
        return
    predictions = load_predictions(args.published_predictions, args.premirbench_predictions)
    assert predictions["score"].notna().all()
    ap_with_exclusions(df, predictions, flags, args.seed, args.draws).to_csv(
        out / "ap_after_overlap_exclusion.tsv", sep="\t", index=False
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
