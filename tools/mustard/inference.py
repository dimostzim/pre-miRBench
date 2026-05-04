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
    p.add_argument("--norm-output","--norm_output",choices=["y", "n"],default="n",
        help="Also write unified_predictions.csv with window_id,probability_score",)
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

    if args.norm_output == "y":
        record_ids = []
        with open(args.targetIntervals) as bed_fh:
            for raw_line in bed_fh:
                parts = raw_line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    record_ids.append(parts[3])

        pattern = os.path.join(
            os.path.abspath(args.dir),
            "predict",
            "static",
            "results",
            "intermediate_files",
            "*.predictions.txt.gz",
        )
        matches = sorted(glob.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one MuStARD predictions file, found {len(matches)} under {pattern}")

        scores = []
        with gzip.open(matches[0], "rt") as gz_fh:
            for raw_line in gz_fh:
                line = raw_line.strip()
                if not line:
                    continue
                cols = [float(value) for value in line.split("\t") if value]
                if len(cols) < 2:
                    raise RuntimeError(f"Expected at least two MuStARD score columns in {matches[0]}")
                scores.append(cols[1])

        if len(scores) != len(record_ids):
            raise RuntimeError(f"MuStARD ID/score count mismatch: {len(record_ids)} IDs vs {len(scores)} scores")

        unified_file = os.path.join(os.path.abspath(args.dir), "unified_predictions.csv")
        with open(unified_file, "w", newline="") as csv_fh:
            writer = csv.writer(csv_fh)
            writer.writerow(["window_id", "probability_score"])
            for record_id, score in zip(record_ids, scores):
                writer.writerow([record_id, score])


if __name__ == "__main__":
    main()
