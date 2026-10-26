#!/usr/bin/env python3
"""
Compare homology-based miRNA annotation calls (MirMachine, ncOrtho) with MirGeneDB loci
and score the genome-scale candidates with them.

Inputs
  --calls          MirMachine per-family GFFs (all hits, *.PRE.gff) or an ncOrtho TSV
                   (columns: human_mirna_id, chicken_contig, start, end, strand, bitscore;
                   1-based inclusive coordinates)
  --called         optional MirMachine *.PRE.filtered.gff files: hits passing MirMachine's
                   family thresholds; without it every call counts as called
  --loci           loci.csv from prepare_genome_candidates.py (MirGeneDB precursors of the
                   genome, with family and the masked flag)
  --candidates     candidates.csv from prepare_genome_candidates.py (optional)
  --training-families  benchmark dataset.csv (families of training positives)

Outputs
  --summary        one row: calls, calls per Mb, precision of called hits (overlapping an
                   annotated precursor on the same strand), and locus recall for the locus
                   groups of genome_scale_metrics.py (all loci, unmasked loci, unmasked loci in
                   training / held-out families)
  --predictions    per-candidate score = best bitscore of any hit overlapping the candidate
                   hairpin on the same strand, plus a called flag; with --candidates the
                   summary also gets AP over unambiguous candidates and the precision of
                   called unambiguous candidates, as in genome_scale_metrics.py
"""
import argparse
import csv
import glob
from bisect import bisect_left
from collections import defaultdict

from sklearn.metrics import average_precision_score

NO_HIT_SCORE = -1000.0


def read_gff(paths):
    hits = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            with open(path) as handle:
                for line in handle:
                    if not line.strip() or line.startswith("#"):
                        continue
                    f = line.rstrip("\n").split("\t")
                    hits.append((f[0], int(f[3]) - 1, int(f[4]), f[6], float(f[5])))
    return hits


def read_ncortho(path):
    hits = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            start, end = sorted((int(row["start"]), int(row["end"])))
            hits.append((row["chicken_contig"], start - 1, end, row["strand"], float(row["bitscore"])))
    return hits


def index(hits):
    by_key = defaultdict(list)
    for chrom, start, end, strand, score in hits:
        by_key[(chrom, strand)].append((start, end, score))
    longest = {key: max(e - s for s, e, _ in items) for key, items in by_key.items()}
    for items in by_key.values():
        items.sort()
    return by_key, longest


def overlapping(by_key, longest, chrom, strand, start, end):
    items = by_key.get((chrom, strand), [])
    first = bisect_left(items, (start - longest.get((chrom, strand), 0), -1, -1e18))
    last = bisect_left(items, (end, -1, -1e18))
    return [item for item in items[first:last] if item[1] > start]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--calls", nargs="+", required=True)
    parser.add_argument("--called", nargs="*", default=None)
    parser.add_argument("--format", choices=("mirmachine_gff", "ncortho_tsv"), required=True)
    parser.add_argument("--loci", required=True)
    parser.add_argument("--training-families", required=True)
    parser.add_argument("--genome-bases", type=int, required=True)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--predictions", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    hits = read_gff(args.calls) if args.format == "mirmachine_gff" else read_ncortho(args.calls[0])
    called_hits = read_gff(args.called) if args.called else hits
    all_index, all_longest = index(hits)
    called_index, called_longest = index(called_hits)

    with open(args.loci, newline="") as handle:
        loci = list(csv.DictReader(handle))
    locus_index, locus_longest = index([(l["chrom"], int(l["start"]), int(l["end"]), l["strand"], 0.0) for l in loci])
    with open(args.training_families, newline="") as handle:
        training = {r["family_id"] for r in csv.DictReader(handle) if r["split"] == "train" and r["label"] == "1"}
    unmasked = [locus for locus in loci if locus["masked"] == "0"]
    locus_groups = {
        "all_loci": loci,
        "unmasked_loci": unmasked,
        "unmasked_training_family_loci": [locus for locus in unmasked if locus["family_id"] in training],
        "unmasked_heldout_family_loci": [locus for locus in unmasked if locus["family_id"] not in training],
    }

    true_calls = sum(1 for c, s, e, st, _ in called_hits if overlapping(locus_index, locus_longest, c, st, s, e))
    summary = {
        "tool": args.tool,
        "calls": len(called_hits),
        "calls_per_mb": len(called_hits) / (args.genome_bases / 1e6),
        "precision_calls_on_annotated_loci": true_calls / len(called_hits) if called_hits else float("nan"),
    }
    for group, members in locus_groups.items():
        recalled = sum(1 for l in members if overlapping(called_index, called_longest, l["chrom"], l["strand"], int(l["start"]), int(l["end"])))
        summary[group] = len(members)
        summary[f"{group}_recall"] = recalled / len(members) if members else float("nan")

    if args.candidates and args.predictions:
        labels, scores, called_flags = [], [], []
        with open(args.candidates, newline="") as handle, open(args.predictions, "w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["tool", "split", "record_id", "species", "label", "score", "called"])
            for row in csv.DictReader(handle):
                start, end = int(row["hairpin_start"]), int(row["hairpin_end"])
                score = max((s for _, _, s in overlapping(all_index, all_longest, row["chrom"], row["strand"], start, end)), default=NO_HIT_SCORE)
                called = bool(overlapping(called_index, called_longest, row["chrom"], row["strand"], start, end))
                writer.writerow([args.tool, row["split"], row["record_id"], row["species"],
                                 1 if row["genome_label"] == "positive" else 0, score, int(called)])
                if row["genome_label"] != "ambiguous":
                    labels.append(row["genome_label"] == "positive")
                    scores.append(score)
                    called_flags.append(called)
        called_labels = [label for label, flag in zip(labels, called_flags) if flag]
        summary["ap_unambiguous_candidates"] = average_precision_score(labels, scores)
        summary["precision_unambiguous_calls"] = sum(called_labels) / len(called_labels) if called_labels else float("nan")

    with open(args.summary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
