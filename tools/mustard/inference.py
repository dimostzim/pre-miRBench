#!/usr/bin/env python
import argparse
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


if __name__ == "__main__":
    main()
