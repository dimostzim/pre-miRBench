#!/usr/bin/env python3
"""Sequence-similarity audit between pre-miRBench test records and training records (blastn, both strands).

Genomic coordinates cannot detect paralogues, orthologues or repeated sequence in
different genomes; this complements overlap_audit.py.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_NAME = {
    "train": "Train",
    "valid": "Validation",
    "test_known_species_known_family": "Test 1",
    "test_known_species_heldout_family": "Test 2",
    "test_heldout_species_known_family": "Test 3",
    "test_heldout_species_heldout_family": "Test 4",
}
TESTS = list(SPLIT_NAME)[2:]
OUTFMT = "6 qseqid sseqid pident length qlen slen qstart qend sstart send evalue bitscore"


def write_fasta(path, ids, sequences):
    with open(path, "w") as handle:
        for record_id, sequence in zip(ids, sequences):
            handle.write(f">{record_id}\n{sequence.replace('U', 'T')}\n")


def run_blast(query, subject_fasta, out, threads, evalue):
    subprocess.run(["makeblastdb", "-in", subject_fasta, "-dbtype", "nucl", "-out", subject_fasta], check=True, capture_output=True)
    subprocess.run(
        [
            "blastn", "-task", "blastn", "-query", query, "-db", subject_fasta, "-strand", "both",
            "-evalue", str(evalue), "-max_target_seqs", "50", "-num_threads", str(threads),
            "-outfmt", OUTFMT, "-out", out,
        ],
        check=True,
    )
    hits = pd.read_csv(out, sep="\t", header=None, names=OUTFMT.split()[1:])
    hits = hits[hits["qseqid"] != hits["sseqid"]]
    hits["query_coverage"] = hits["length"] / hits["qlen"]
    return hits


def best_hits(hits):
    return hits.sort_values(["qseqid", "bitscore"], ascending=[True, False]).drop_duplicates("qseqid")


def precursor_level(df, work, threads, evalue):
    """Test-positive precursors against training-positive precursors."""
    positives = df[df["label"] == 1]
    train = positives[positives["split"] == "train"]
    tests = positives[positives["split"].isin(TESTS)]
    write_fasta(work / "train_precursors.fa", train["record_id"], train["precursor_sequence"])
    write_fasta(work / "test_precursors.fa", tests["record_id"], tests["precursor_sequence"])
    hits = run_blast(work / "test_precursors.fa", str(work / "train_precursors.fa"), work / "precursor_hits.tsv", threads, evalue)
    meta = df.set_index("record_id")[["split", "species", "family_id", "mirna_id"]]
    hits = hits.join(meta.add_prefix("q_"), on="qseqid").join(meta.add_prefix("s_"), on="sseqid")
    hits["same_family_name"] = hits["q_family_id"] == hits["s_family_id"]

    # A test precursor is called "near-identical to training" when one alignment spans
    # at least 80% of the test precursor at >=90% identity.
    near = hits[(hits["pident"] >= 90) & (hits["query_coverage"] >= 0.8)]
    similar = hits[(hits["pident"] >= 80) & (hits["query_coverage"] >= 0.5)]
    rows = []
    for split in TESTS:
        q = tests[tests["split"] == split]
        ids = set(q["record_id"])
        for label, table in (("near_identical_ge90pid_ge80cov", near), ("similar_ge80pid_ge50cov", similar), ("any_blast_hit_e1e-5", hits)):
            t = table[table["qseqid"].isin(ids)]
            rows.append(
                {
                    "test": SPLIT_NAME[split],
                    "criterion": label,
                    "test_positives": len(q),
                    "with_training_match": t["qseqid"].nunique(),
                    "with_training_match_same_family_name": t.loc[t["same_family_name"], "qseqid"].nunique(),
                    "with_training_match_only_other_family_names": len(
                        set(t["qseqid"]) - set(t.loc[t["same_family_name"], "qseqid"])
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    summary["fraction_with_training_match"] = summary["with_training_match"] / summary["test_positives"]

    best = best_hits(hits)
    cross_family = best_hits(near[~near["same_family_name"] & near["q_split"].str.endswith("heldout_family")])
    species_of_best = (
        best_hits(near)[["q_split", "q_species", "s_species"]]
        .value_counts()
        .rename("test_positives")
        .reset_index()
        .query("q_split.str.startswith('test_heldout_species')", engine="python")
    )
    return summary, best, cross_family, species_of_best


def window_level(df, work, threads, evalue):
    """All test windows against all training windows."""
    train = df[df["split"] == "train"]
    tests = df[df["split"].isin(TESTS)]
    write_fasta(work / "train_windows.fa", train["record_id"], train["sequence"])
    write_fasta(work / "test_windows.fa", tests["record_id"], tests["sequence"])
    hits = run_blast(work / "test_windows.fa", str(work / "train_windows.fa"), work / "window_hits.tsv", threads, evalue)
    label = df.set_index("record_id")["label"]
    hits["q_label"] = hits["qseqid"].map(label)
    hits["s_label"] = hits["sseqid"].map(label)
    best = best_hits(hits)
    rows = []
    for split in TESTS:
        q = tests[tests["split"] == split]
        for q_label in (1, 0):
            ids = set(q.loc[q["label"] == q_label, "record_id"])
            b = best[best["qseqid"].isin(ids)]
            strong = hits[hits["qseqid"].isin(ids) & (hits["pident"] >= 90) & (hits["length"] >= 100)]
            rows.append(
                {
                    "test": SPLIT_NAME[split],
                    "label": "pos" if q_label else "neg",
                    "records": len(ids),
                    "any_hit": b["qseqid"].nunique(),
                    "hit_ge90pid_ge100nt": strong["qseqid"].nunique(),
                    "hit_ge90pid_ge100nt_to_same_label": strong.loc[strong["s_label"] == q_label, "qseqid"].nunique(),
                    "hit_ge90pid_ge100nt_to_opposite_label": strong.loc[strong["s_label"] != q_label, "qseqid"].nunique(),
                }
            )
    return pd.DataFrame(rows), best


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--evalue", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    work = out / "blast_work"
    work.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset, keep_default_na=False)
    df["label"] = df["label"].astype(int)

    summary, best, cross_family, species_of_best = precursor_level(df, work, args.threads, args.evalue)
    summary.to_csv(out / "precursor_similarity_to_train_summary.tsv", sep="\t", index=False)
    best.to_csv(out / "precursor_best_training_hit.tsv", sep="\t", index=False)
    cross_family.to_csv(out / "heldout_family_precursors_near_identical_to_other_training_family.tsv", sep="\t", index=False)
    species_of_best.to_csv(out / "heldout_species_best_hit_species.tsv", sep="\t", index=False)

    window_summary, window_best = window_level(df, work, args.threads, args.evalue)
    window_summary.to_csv(out / "window_similarity_to_train_summary.tsv", sep="\t", index=False)
    window_best.to_csv(out / "window_best_training_hit.tsv.gz", sep="\t", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
