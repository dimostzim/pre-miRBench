#!/usr/bin/env python3
"""
Label genome-scale hairpin candidates and write predictor inputs.

Candidates come from `extract_hairpin_candidates.py --mode full` run on a held-out
genome, without excluding annotated loci. Labels:

  positive   same strand as an annotated precursor and terminal-loop centre
             (the window centre) within --tolerance nt of the precursor midpoint
             (the closest such candidate per precursor)
  ambiguous  any other candidate whose window overlaps an annotated precursor
             (either strand, plus --buffer); excluded from precision-recall
             metrics but counted as calls
  negative   every other candidate

Writes, under --output-dir:
  candidates.csv               one row per candidate with label and provenance
  loci.csv                     one row per annotated precursor with its matched candidate and
                               whether the precursor fails the repeat rule applied to the
                               hairpins of benchmark positives (masked)
  tool_inputs/<tool>/          FASTA/BED inputs for the published predictors, split
                               genome_<species>_{positive,negative} (ambiguous
                               candidates are written with the negatives; use
                               candidates.csv labels for metrics)
  premirbench_inputs/genome_<species>/input/samples.tsv and labels.csv
"""
import argparse
import csv
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset"))
from build_dataset import family_id_from_mirna_id  # noqa: E402
from prepare_tool_inputs import DEFAULT_TARGET_LENGTHS, TOOLS, prepare_row, write_bed, write_fasta  # noqa: E402
from extract_hairpin_candidates import count_masked, load_fasta  # noqa: E402


def read_precursors(bed):
    precursors = []
    with open(bed) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 6 and parts[3].endswith("_pre"):
                precursors.append({"chrom": parts[0], "start": int(parts[1]), "end": int(parts[2]), "name": parts[3], "strand": parts[5]})
    return precursors


def label_candidates(candidates, precursors, tolerance, buffer):
    by_strand = defaultdict(list)
    for index, row in enumerate(candidates):
        midpoint = (int(row["hairpin_loop_start"]) + int(row["hairpin_loop_end"])) / 2
        by_strand[(row["chrom"], row["strand"])].append((midpoint, index))
    for key in by_strand:
        by_strand[key].sort()

    labels = ["negative"] * len(candidates)
    loci = []
    for precursor in precursors:
        target = (precursor["start"] + precursor["end"]) / 2
        pool = by_strand.get((precursor["chrom"], precursor["strand"]), [])
        keys = [midpoint for midpoint, _ in pool]
        hits = pool[bisect_left(keys, target - tolerance):bisect_right(keys, target + tolerance)]
        matched = min(hits, key=lambda item: abs(item[0] - target))[1] if hits else None
        if matched is not None:
            labels[matched] = "positive"
            candidates[matched]["mirna_id"] = precursor["name"]
            candidates[matched]["family_id"] = family_id_from_mirna_id(precursor["name"])
            candidates[matched]["target_start"] = str(precursor["start"])
            candidates[matched]["target_end"] = str(precursor["end"])
        loci.append({**precursor, "family_id": family_id_from_mirna_id(precursor["name"]),
                     "matched_record_id": "" if matched is None else candidates[matched]["record_id"]})

    by_chrom = defaultdict(list)
    for precursor in precursors:
        by_chrom[precursor["chrom"]].append((precursor["start"] - buffer, precursor["end"] + buffer))
    for intervals in by_chrom.values():
        intervals.sort()
    for index, row in enumerate(candidates):
        if labels[index] == "positive":
            continue
        start, end = int(row["start"]), int(row["end"])
        intervals = by_chrom.get(row["chrom"], [])
        position = bisect_left(intervals, (end, end))
        if any(s < end and start < e for s, e in intervals[max(0, position - 50):position]):
            labels[index] = "ambiguous"
    return labels, loci


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--bed", required=True, help="Prefixed precursor BED for the same genome.")
    parser.add_argument("--species", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tolerance", type=int, default=15)
    parser.add_argument("--buffer", type=int, default=100)
    parser.add_argument("--genome", required=True, help="Prefixed genome FASTA, for the masked flag of each locus.")
    parser.add_argument("--max-repeat-frac", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = f"genome_{args.species}"

    with open(args.candidates, newline="") as handle:
        candidates = list(csv.DictReader(handle))
    for index, row in enumerate(candidates, start=1):
        row["record_id"] = f"{args.species}_genome_{index:08d}"
        row["species"] = args.species
        row["split"] = split
        row["mirna_id"] = row["family_id"] = ""
        # Crops for 100/160/180-nt predictors keep the hairpin, for every candidate alike.
        row["target_start"], row["target_end"] = row["hairpin_start"], row["hairpin_end"]
    labels, loci = label_candidates(candidates, read_precursors(args.bed), args.tolerance, args.buffer)
    genome = load_fasta(args.genome)
    for locus in loci:
        # The repeat rule of benchmark positives (extract_hairpin_candidates.fails_repeat_rule), applied to the precursor.
        precursor = genome.get(locus["chrom"], "")[locus["start"]:locus["end"]]
        locus["masked"] = int(not precursor or count_masked(precursor) / len(precursor) > args.max_repeat_frac)
    del genome
    for row, label in zip(candidates, labels):
        row["genome_label"] = label
        row["label"] = "positive" if label == "positive" else "negative"

    fields = ["record_id", "species", "split", "window_id", "chrom", "start", "end", "strand", "sequence", "structure", "mfe",
              "hairpin_start", "hairpin_end", "hairpin_loop_start", "hairpin_loop_end", "hairpin_mfe", "target_start", "target_end", "mirna_id", "family_id", "genome_label"]
    with open(out / "candidates.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)
    with open(out / "loci.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chrom", "start", "end", "name", "strand", "family_id", "masked", "matched_record_id"])
        writer.writeheader()
        writer.writerows(loci)

    for tool in TOOLS:
        tool_dir = out / "tool_inputs" / tool
        tool_dir.mkdir(parents=True, exist_ok=True)
        prepared = [prepare_row(row, tool, DEFAULT_TARGET_LENGTHS.get(tool)) for row in candidates]
        for label in ("positive", "negative"):
            rows = [row for row in prepared if row["label"] == label]
            write_fasta(tool_dir / f"{split}_{label}.fa", rows)
            write_bed(tool_dir / f"{split}_{label}.bed", rows)

    model_dir = out / "premirbench_inputs" / split / "input"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "samples.tsv", "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "species", "sequence_rna", "structure", "mfe"])
        for row in candidates:
            writer.writerow([row["record_id"], row["species"], row["sequence"], row["structure"], row["mfe"]])
    with open(model_dir.parent / "labels.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        for row in candidates:
            writer.writerow([row["record_id"], 1 if row["genome_label"] == "positive" else 0])

    counts = {label: labels.count(label) for label in ("positive", "ambiguous", "negative")}
    matched = sum(1 for locus in loci if locus["matched_record_id"])
    unmasked = [locus for locus in loci if not locus["masked"]]
    matched_unmasked = sum(1 for locus in unmasked if locus["matched_record_id"])
    print(f"{args.species}: candidates={len(candidates)} {counts} loci_matched={matched}/{len(loci)} "
          f"unmasked_loci_matched={matched_unmasked}/{len(unmasked)}")


if __name__ == "__main__":
    main()
