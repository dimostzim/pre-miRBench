#!/usr/bin/env python
import argparse
import csv
import glob
import gzip
import os
import subprocess


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--targetIntervals", required=True)
    p.add_argument("--genome", required=True)
    p.add_argument("--consDir", required=True)
    
    p.add_argument("--chromList", default="all")
    p.add_argument("--dir", default="results")
    p.add_argument("--model", default="MuStARD-mirSFC-U")
    p.add_argument("--classNum", type=int, default=2)
    p.add_argument("--modelType", default="CNN")
    
    # optional
    p.add_argument("--modelDirName", default="results")
    p.add_argument("--intermDir", default="same")
    p.add_argument("--winSize", type=int, default=100)
    p.add_argument("--staticPredFlag", type=int, default=0)
    p.add_argument("--inputMode", default="sequence,RNAfold,conservation")  # best model uses all 3 sequence types
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--step", type=int, default=5)
    args = p.parse_args()

    perl_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mustard_src", "MuStARD.pl")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # resolve model name to full path 
    model_path = os.path.join(base_dir, "data", "models", args.model, "CNNonRaw.hdf5")
    args.model = model_path

    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    cmd = [
        "perl",
        perl_script,
        "predict",
        "--chromList", args.chromList,
        "--targetIntervals", os.path.abspath(args.targetIntervals),
        "--genome", os.path.abspath(args.genome),
        "--consDir", os.path.abspath(args.consDir),
        "--dir", os.path.abspath(args.dir),
        "--model", os.path.abspath(args.model),
        "--classNum", str(args.classNum),
        "--modelType", args.modelType,
        "--winSize", str(args.winSize),
        "--step", str(args.step),
        "--staticPredFlag", str(args.staticPredFlag),
        "--inputMode", args.inputMode,
        "--threads", str(args.threads),
        "--modelDirName", args.modelDirName,
    ]
    
    if args.intermDir != "same":
        cmd.extend(["--intermDir", os.path.abspath(args.intermDir)])

    subprocess.check_call(cmd)

    # --- Post-processing: write unified predictions.csv ---
    # Read record IDs grouped by chromosome (preserving per-chrom order from BED).
    chrom_records = {}  # chrom -> list of record_ids in BED order
    all_record_ids = []  # global order from BED
    chrom_order_global = {}  # record_id -> global index
    with open(os.path.abspath(args.targetIntervals)) as bed_fh:
        for line in bed_fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                chrom = parts[0]
                rid = parts[3]
                chrom_records.setdefault(chrom, []).append(rid)
                chrom_order_global[rid] = len(all_record_ids)
                all_record_ids.append(rid)

    # Build a score dict: record_id -> score, reading per-chromosome prediction files.
    pred_base = os.path.join(os.path.abspath(args.dir),
                             "predict", "static", "results", "intermediate_files")
    scores_by_id = {}
    for chrom, chrom_ids in chrom_records.items():
        gz_path = os.path.join(pred_base, f"targets.{chrom}.predictions.txt.gz")
        chrom_scores = []
        with gzip.open(gz_path, "rt") as gz_fh:
            for raw_line in gz_fh:
                line = raw_line.strip()
                if line:
                    cols = [float(v) for v in line.split("\t") if v]
                    chrom_scores.append(cols[1])
        if len(chrom_scores) != len(chrom_ids):
            raise RuntimeError(
                f"mustard ID/score count mismatch on {chrom}: "
                f"{len(chrom_ids)} IDs vs {len(chrom_scores)} scores"
            )
        for rid, score in zip(chrom_ids, chrom_scores):
            scores_by_id[rid] = score

    csv_path = os.path.join(os.path.abspath(args.dir), "predictions.csv")
    with open(csv_path, "w", newline="") as csv_fh:
        writer = csv.writer(csv_fh)
        writer.writerow(["window_id", "probability_score"])
        for rid in all_record_ids:
            writer.writerow([rid, scores_by_id[rid]])


if __name__ == "__main__":
    main()
