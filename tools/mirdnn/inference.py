#!/usr/bin/env python
import argparse
import csv
import os
import subprocess
import sys
import tempfile


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True, help="Input FASTA file")
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--model", default="animal", choices=["animal", "plants"], help="Pre-trained model to use")
    p.add_argument("--seq_length", type=int, default=160, help="Sequence length for padding/truncation")
    p.add_argument("--device", default="cpu", help="Device to use (cpu, cuda:0, etc.)")
    p.add_argument("--batch_size", type=int, default=1024, help="Batch size for inference")
    p.add_argument("--input_fold", default=None, help="Precomputed RNAfold output (skip RNAfold)")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Prefer bundled source inside the Docker image if present.
    mirdnn_src = "/opt/mirdnn/mirdnn_src" if os.path.isdir("/opt/mirdnn/mirdnn_src") else os.path.join(base_dir, "mirdnn_src")
    conda_bin = os.path.dirname(sys.executable)

    os.makedirs(args.output, exist_ok=True)

    if args.input_fold:
        fold_file = os.path.abspath(args.input_fold)
    else:
        # Remove old fold file to prevent RNAfold from appending across runs
        fold_file = os.path.join(os.path.abspath(args.output), "input.fold")
        if os.path.exists(fold_file):
            os.remove(fold_file)
        rnafold_cmd = [
            os.path.join(conda_bin, "RNAfold"),
            "--noPS",
            f"--infile={os.path.abspath(args.input)}",
            "--outfile=input.fold",
        ]
        subprocess.check_call(rnafold_cmd, cwd=os.path.abspath(args.output))

    model_path = os.path.join(mirdnn_src, "models", f"{args.model}.pmt")
    eval_script = os.path.join(mirdnn_src, "mirdnn_eval.py")
    output_file = os.path.join(os.path.abspath(args.output), "predictions.csv")
    # Remove any previous predictions to prevent the upstream from appending to it
    if os.path.exists(output_file):
        os.remove(output_file)

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

    # Add header row to the headerless CSV produced by the upstream script
    with open(output_file, newline="") as f:
        rows = list(csv.reader(f))
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_id", "probability_score"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
