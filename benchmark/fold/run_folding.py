#!/usr/bin/env python3
import argparse
import os
import subprocess

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "output")

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "fold_output"))
parser.add_argument("--window", type=int, default=200)
parser.add_argument("--step", type=int, default=50)
parser.add_argument("--chr", dest="chromosomes", default=None,
                    help="Comma-separated list of chromosomes to process (default: all)")
parser.add_argument("--cpus", type=int, default=8)
parser.add_argument("--dna", action="store_true")
parser.add_argument("--single_strand", action="store_true", help="Only forward strand (default: both strands)")
parser.add_argument("--max_repeat_frac", type=float, default=0.1,
                    help="Max fraction of masked bases to allow (default: 0.1)")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

cmd = ["python", os.path.join(os.path.dirname(__file__), "make_windows.py"), "--input", args.input, "--window", str(args.window), "--step", str(args.step), "--output", f"{args.output}/windows.fa"]
if args.chromosomes:
    cmd += ["--chr", args.chromosomes]
if args.dna:
    cmd.append("--dna")
if args.single_strand:
    cmd.append("--single_strand")
cmd += ["--max_repeat_frac", str(args.max_repeat_frac)]
subprocess.check_call(cmd)

cmd = ["python", os.path.join(os.path.dirname(__file__), "run_fold.py"), "--input", f"{args.output}/windows.fa", "--output", f"{args.output}/windows.fold"]
if args.cpus:
    cmd += ["--cpus", str(args.cpus)]
subprocess.check_call(cmd)

subprocess.check_call(["python", os.path.join(os.path.dirname(__file__), "analyze_fold.py"), "--input", f"{args.output}/windows.fold", "--csv", f"{args.output}/results.csv", "--plot", f"{args.output}/mfe_kde.png"])
