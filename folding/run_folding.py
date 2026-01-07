#!/usr/bin/env python3
import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--window", type=int, default=200)
parser.add_argument("--step", type=int, default=50)
parser.add_argument("--cpus", type=int, default=None)
parser.add_argument("--dna", action="store_true")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

cmd = ["python", "scripts/make_windows.py", "--input", args.input, "--window", str(args.window), "--step", str(args.step), "--output", f"{args.output}/windows.fa"]
if args.dna:
    cmd.append("--dna")
subprocess.check_call(cmd)

cmd = ["python", "scripts/run_fold.py", "--input", f"{args.output}/windows.fa", "--output", f"{args.output}/windows.fold"]
if args.cpus:
    cmd += ["--cpus", str(args.cpus)]
subprocess.check_call(cmd)

subprocess.check_call(["python", "scripts/analyze_fold.py", "--input", f"{args.output}/windows.fold", "--csv", f"{args.output}/results.csv", "--plot", f"{args.output}/mfe_kde.png"])
