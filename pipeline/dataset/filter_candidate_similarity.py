#!/usr/bin/env python3
"""
Remove candidate negative windows that resemble annotated precursors or each other.

Runs once over the candidate pools of all species, before hard-negative mining:

1. Windows containing a near-copy of any annotated precursor (blastn, at least
   --precursor-identity % identity over at least --precursor-coverage of the
   precursor length) are removed. Such windows may be unannotated paralogues or
   orthologues of known miRNAs.
2. Remaining windows and their reverse complements are clustered with MMseqs2
   (easy-cluster, at least --window-identity identity and --window-coverage
   coverage of both sequences). Windows whose forward or reverse-complement copies
   share a cluster are linked; one window per linked group is kept, so
   near-identical sequence on either strand cannot appear in two partitions.

Writes one filtered pool CSV per input pool and a per-species report.
"""
import argparse
import csv
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

OUTFMT = "6 qseqid sseqid pident length qlen slen"
COMPLEMENT = str.maketrans("ACGUTN", "UGCAAN")


def reverse_complement(sequence):
    return sequence.upper().translate(COMPLEMENT)[::-1]


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def blast(query, subject, out, task, threads, extra=()):
    subprocess.run(["makeblastdb", "-in", subject, "-dbtype", "nucl", "-out", subject], check=True, capture_output=True)
    subprocess.run(
        ["blastn", "-task", task, "-query", query, "-db", subject, "-strand", "both", "-evalue", "1e-5",
         "-num_threads", str(threads), "-outfmt", OUTFMT, "-out", out, *extra],
        check=True,
    )
    with open(out) as handle:
        for line in handle:
            qseqid, sseqid, pident, length, qlen, slen = line.split("\t")
            yield qseqid, sseqid, float(pident), int(length), int(qlen), int(slen)


def write_fasta(path, records):
    with open(path, "w") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n{sequence.replace('U', 'T')}\n")


def clusters(nodes, edges):
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first, second in edges:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[max(root_first, root_second)] = min(root_first, root_second)
    groups = defaultdict(list)
    for node in nodes:
        groups[find(node)].append(node)
    return groups.values()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="TSV with columns species, pool_csv, filtered_csv.")
    parser.add_argument("--precursor-fasta", required=True, nargs="+", help="FASTA files of all annotated precursors.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--precursor-identity", type=float, default=90.0)
    parser.add_argument("--precursor-coverage", type=float, default=0.5)
    parser.add_argument("--window-identity", type=float, default=0.90)
    parser.add_argument("--window-coverage", type=float, default=0.50)
    return parser.parse_args()


def main():
    args = parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))

    pools = {}
    order = {}
    for species_order, entry in enumerate(manifest):
        rows = read_rows(entry["pool_csv"])
        pools[entry["species"]] = rows
        for row in rows:
            order[row["window_id"]] = (species_order, row["window_id"])
    windows_fasta = work / "candidate_windows.fa"
    write_fasta(windows_fasta, ((row["window_id"], row["sequence"]) for rows in pools.values() for row in rows))

    precursor_fasta = work / "precursors.fa"
    with open(precursor_fasta, "w") as out:
        for path in args.precursor_fasta:
            with open(path) as handle:
                out.write(handle.read().replace("U", "T"))

    precursor_similar = set()
    for qseqid, _sseqid, pident, length, _qlen, slen in blast(
        windows_fasta, str(precursor_fasta), work / "windows_vs_precursors.tsv", "blastn", args.threads
    ):
        if pident >= args.precursor_identity and length >= args.precursor_coverage * slen:
            precursor_similar.add(qseqid)

    remaining = [name for name in order if name not in precursor_similar]
    sequences = {row["window_id"]: row["sequence"] for rows in pools.values() for row in rows}
    stranded_fasta = work / "candidate_windows_both_strands.fa"
    write_fasta(
        stranded_fasta,
        ((f"{name}#{strand}", sequence if strand == "f" else reverse_complement(sequence))
         for name in remaining for sequence in [sequences[name]] for strand in ("f", "r")),
    )
    subprocess.run(
        ["mmseqs", "easy-cluster", str(stranded_fasta), str(work / "mmseqs"), str(work / "mmseqs_tmp"),
         "--min-seq-id", str(args.window_identity), "-c", str(args.window_coverage), "--cov-mode", "0",
         "--dbtype", "2", "--threads", str(args.threads), "-v", "1"],
        check=True,
    )
    edges = []
    with open(work / "mmseqs_cluster.tsv") as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")
            first, second = representative.rsplit("#", 1)[0], member.rsplit("#", 1)[0]
            if first != second:
                edges.append((first, second))
    duplicate = set()
    for group in clusters(remaining, edges):
        if len(group) > 1:
            keep = min(group, key=order.get)
            duplicate.update(name for name in group if name != keep)

    report = []
    for entry in manifest:
        rows = pools[entry["species"]]
        kept = [row for row in rows if row["window_id"] not in precursor_similar and row["window_id"] not in duplicate]
        os.makedirs(os.path.dirname(os.path.abspath(entry["filtered_csv"])), exist_ok=True)
        with open(entry["filtered_csv"], "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["window_id"])
            writer.writeheader()
            writer.writerows(kept)
        counts = Counter(
            "precursor_similar" if row["window_id"] in precursor_similar else "near_duplicate" if row["window_id"] in duplicate else "kept"
            for row in rows
        )
        report.append({"species": entry["species"], "candidates": len(rows), **{key: counts[key] for key in ("precursor_similar", "near_duplicate", "kept")}})

    with open(args.report, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["species", "candidates", "precursor_similar", "near_duplicate", "kept"])
        writer.writeheader()
        writer.writerows(report)
    total = Counter()
    for row in report:
        total.update({key: row[key] for key in ("candidates", "precursor_similar", "near_duplicate", "kept")})
    print(f"candidates: {total['candidates']} precursor_similar: {total['precursor_similar']} "
          f"near_duplicate: {total['near_duplicate']} kept: {total['kept']}")


if __name__ == "__main__":
    main()
