#!/usr/bin/env python3
"""pre-miRBench combined with retrained MirMachine: one score per record (user, 29 Oct 2026).

Combined score = pre-miRBench score + 1 when MirMachine calls the record (a hit at or above its family's trusted
cutoff), otherwise the pre-miRBench score. MirMachine's calls therefore rank first, in pre-miRBench order, and every
other record follows in pre-miRBench order. The rule has no parameter; MirMachine's cutoffs were fitted on the training
genomes. Writes the columns of the predictors' prediction files (tool, split, record_id, species, label, score) for one
split: the validation partition (for the thresholds) or one genome (genome_<sp>).
"""
import argparse
import csv

TOOL = "premirbench_mirmachine_combined"


def read(path, split):
    with open(path, newline="") as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle) if row["split"] == split}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--premirbench", required=True, help="pre-miRBench predictions.csv")
    parser.add_argument("--mirmachine", required=True, help="MirMachine predictions with a called column")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    premirbench, mirmachine = read(args.premirbench, args.split), read(args.mirmachine, args.split)
    if not premirbench or premirbench.keys() != mirmachine.keys():
        raise SystemExit(f"{args.split}: record ids differ between {args.premirbench} and {args.mirmachine}")
    called = 0
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tool", "split", "record_id", "species", "label", "score"])
        for record_id, row in premirbench.items():
            other, score = mirmachine[record_id], float(row["score"])
            if other["label"] != row["label"]:
                raise SystemExit(f"{record_id}: labels differ")
            if not 0.0 <= score <= 1.0:
                raise SystemExit(f"{record_id}: pre-miRBench score {score} outside [0, 1]")
            called += int(other["called"])
            writer.writerow([TOOL, args.split, record_id, row["species"], row["label"], score + int(other["called"])])
    print(f"{args.split}: {len(premirbench)} records, {called} called by MirMachine -> {args.output}")


if __name__ == "__main__":
    main()
