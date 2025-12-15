#!/usr/bin/env python
import argparse
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True, help="Input FASTA file")
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--model", default="animal", choices=["animal", "plants"], help="Pre-trained model to use")
    p.add_argument("--seq_length", type=int, default=160, help="Sequence length for padding/truncation")
    p.add_argument("--device", default="cpu", help="Device to use (cpu, cuda:0, etc.)")
    p.add_argument("--batch_size", type=int, default=1024, help="Batch size for inference")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    mirdnn_src = os.path.join(base_dir, "mirdnn_src")
    conda_bin = os.path.dirname(sys.executable)

    os.makedirs(args.output, exist_ok=True)

    rnafold_cmd = [
        os.path.join(conda_bin, "RNAfold"),
        "--noPS",
        f"--infile={os.path.abspath(args.input)}",
        "--outfile=input.fold"
    ]
    subprocess.check_call(rnafold_cmd, cwd=os.path.abspath(args.output))

    fold_file = os.path.join(args.output, "input.fold")

    model_path = os.path.join(mirdnn_src, "models", f"{args.model}.pmt")
    eval_script = os.path.join(mirdnn_src, "mirdnn_eval.py")
    output_file = os.path.join(os.path.abspath(args.output), "predictions.csv")

    cmd = [
        sys.executable,
        eval_script,
        "-i", os.path.abspath(fold_file),
        "-o", output_file,
        "-m", model_path,
        "-s", str(args.seq_length),
        "-d", args.device,
        "-b", str(args.batch_size)
    ]

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
