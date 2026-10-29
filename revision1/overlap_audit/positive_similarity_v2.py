#!/usr/bin/env python3
"""
Sequence similarity between v2 positives of different partitions (user request 22 Oct): for every validation and test
positive, is a training positive nearly identical? Same tool and thresholds as the build's similarity filter
(pipeline/dataset/filter_candidate_similarity.py: blastn -task blastn, both strands, E-value 1e-5, at least 90%
identity), applied twice:
  precursor  test precursor against training precursors; alignment covering at least 50% of the test precursor
  window     test window against training windows; alignment covering at least 50% of both 200-nt windows
A match is labelled same_family or other_family by the families of the two positives. Known-family tests share
families with training by design; for the held-out-family tests (2 and 4) any match is to another family.
usage: positive_similarity_v2.py <dataset.csv> <work dir> <output dir>
"""
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

dataset, work, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
work.mkdir(parents=True, exist_ok=False)
rows = [row for row in csv.DictReader(open(dataset, newline="")) if row["label"] == "1"]
train = [row for row in rows if row["split"] == "train"]
others = [row for row in rows if row["split"] != "train"]
family = {row["record_id"]: row["family_id"] for row in rows}
split = {row["record_id"]: row["split"] for row in rows}


def fasta(path, records, field):
    with open(path, "w") as handle:
        for row in records:
            handle.write(f">{row['record_id']}\n{row[field].upper().replace('U', 'T')}\n")


matches = defaultdict(dict)  # (kind, query) -> best match per subject family relation
details = []
for kind, field in (("precursor", "precursor_sequence"), ("window", "sequence")):
    subject, query, hits = work / f"train_{kind}.fa", work / f"other_{kind}.fa", work / f"{kind}.tsv"
    fasta(subject, train, field)
    fasta(query, others, field)
    subprocess.run(["makeblastdb", "-in", subject, "-dbtype", "nucl", "-out", subject], check=True, capture_output=True)
    subprocess.run(["blastn", "-task", "blastn", "-query", query, "-db", subject, "-strand", "both", "-evalue", "1e-5",
                    "-num_threads", "16", "-max_target_seqs", "1000", "-outfmt", "6 qseqid sseqid pident length qlen slen",
                    "-out", hits], check=True)
    for line in open(hits):
        qseqid, sseqid, pident, length, qlen, slen = line.split("\t")
        pident, length, qlen, slen = float(pident), int(length), int(qlen), int(slen)
        covered = length >= 0.5 * qlen if kind == "precursor" else (length >= 0.5 * qlen and length >= 0.5 * slen)
        if pident < 90 or not covered:
            continue
        relation = "same_family" if family[qseqid] == family[sseqid] else "other_family"
        matches[(kind, qseqid)][relation] = True
        details.append({"kind": kind, "record_id": qseqid, "split": split[qseqid], "family_id": family[qseqid],
                        "train_record_id": sseqid, "train_family_id": family[sseqid], "relation": relation,
                        "identity": pident, "alignment_length": length, "query_length": qlen, "train_length": slen})

summary = []
for name in sorted({row["split"] for row in others}):
    members = [row["record_id"] for row in others if row["split"] == name]
    for kind in ("precursor", "window"):
        found = [matches.get((kind, record), {}) for record in members]
        summary.append({"split": name, "kind": kind, "positives": len(members),
                        "with_any_match": sum(bool(m) for m in found),
                        "with_same_family_match": sum("same_family" in m for m in found),
                        "with_other_family_match": sum("other_family" in m for m in found)})
for name, data in (("positive_similarity_summary.tsv", summary), ("positive_similarity_matches.tsv", details)):
    with open(out / name, "w", newline="") as handle:
        fields = list(data[0]) if data else ["kind", "record_id"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(data)
for row in summary:
    print("\t".join(str(value) for value in row.values()))
