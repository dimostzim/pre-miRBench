#!/usr/bin/env python
import argparse
import os
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--genome", required=True)
    p.add_argument("--output", default="results")
    p.add_argument("--model", default="MuStARD-mirSFC-U")
    p.add_argument("--cons", required=True)
    p.add_argument("--chrom", default="all")
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--win-size", type=int, default=100)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--static-pred-flag", type=int, default=0)
    p.add_argument("--model-type", default="CNN")
    p.add_argument("--class-num", default="2")
    args = p.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(base, "data", "models", args.model, "CNNonRaw.hdf5")
    perl_script = os.path.join(base, "mustard_src", "MuStARD.pl")

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    input_mode = {
        "MuStARD-mirS": "sequence",
        "MuStARD-mirF": "RNAfold",
        "MuStARD-mirSF": "sequence,RNAfold",
        "MuStARD-mirSC": "sequence,conservation",
        "MuStARD-mirFC": "RNAfold,conservation",
    }.get(args.model, "sequence,RNAfold,conservation")

    cmd = [
        "perl",
        perl_script,
        "predict",
        "--winSize",
        str(args.win_size),
        "--step",
        str(args.step),
        "--staticPredFlag",
        str(args.static_pred_flag),
        "--modelType",
        args.model_type,
        "--classNum",
        str(args.class_num),
        "--model",
        model_file,
        "--inputMode",
        input_mode,
        "--targetIntervals",
        os.path.abspath(args.input),
        "--genome",
        os.path.abspath(args.genome),
        "--dir",
        os.path.abspath(args.output),
        "--modelDirName",
        "results",
        "--threads",
        str(args.threads),
    ]

    if args.cons:
        cmd.extend(["--consDir", os.path.abspath(args.cons)])
    if args.chrom:
        cmd.extend(["--chromList", args.chrom])

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
