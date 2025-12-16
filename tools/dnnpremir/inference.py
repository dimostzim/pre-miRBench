#!/usr/bin/env python
import argparse
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True, help="Input FASTA file")
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--seq_length", type=int, default=180, help="Sequence length (fixed at 180, for documentation only)")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dnnpremir_src = os.path.join(base_dir, "dnnpremir_src")

    os.makedirs(args.output, exist_ok=True)

    # Output file path
    output_file = os.path.join(os.path.abspath(args.output), "predictions.txt")

    # The original script expects to be run from its own directory
    # because it uses relative paths like ./bin/RNAfold and src/CNN/CNN_model.h5
    cmd = [
        sys.executable,
        "isPreMiR.py",
        "-i", os.path.abspath(args.input),
        "-o", output_file
    ]

    # Run from dnnpremir_src directory so relative paths work
    subprocess.check_call(cmd, cwd=dnnpremir_src)


if __name__ == "__main__":
    main()
