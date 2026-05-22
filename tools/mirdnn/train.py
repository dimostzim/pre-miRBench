#!/usr/bin/env python
import argparse
import os
import subprocess
import sys

import torch as tr


def require_torch_gpu(device):
    if not str(device).startswith("cuda"):
        raise SystemExit(f"mirDNN training requires a CUDA device; got --device {device}")
    if not tr.cuda.is_available():
        raise SystemExit("mirDNN training requires a visible PyTorch CUDA GPU")


def fold_fasta(fasta_path, output_path):
    conda_bin = os.path.dirname(sys.executable)
    cmd = [os.path.join(conda_bin, "RNAfold"), "--noPS"]
    with open(fasta_path) as input_handle, open(output_path, "w") as output_handle:
        subprocess.check_call(cmd, stdin=input_handle, stdout=output_handle)


def main():
    parser = argparse.ArgumentParser(description="Train a mirDNN model.")
    parser.add_argument("--positive_fasta", required=True)
    parser.add_argument("--negative_fasta", required=True)
    parser.add_argument("--validation_positive_fasta")
    parser.add_argument("--validation_negative_fasta")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seq_length", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--early_stop", type=int, default=100)
    parser.add_argument("--valid_prop", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--upsample")
    parser.add_argument("--focal_loss")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    require_torch_gpu(args.device)
    negative_fold = os.path.join(args.output, "negative.fold")
    positive_fold = os.path.join(args.output, "positive.fold")
    fold_fasta(args.negative_fasta, negative_fold)
    fold_fasta(args.positive_fasta, positive_fold)

    mirdnn_src = "/opt/mirdnn/mirdnn_src" if os.path.isdir("/opt/mirdnn/mirdnn_src") else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mirdnn_src"
    )
    fit_script = os.path.join(mirdnn_src, "mirdnn_fit.py")
    model_path = os.path.join(args.output, "model.pmt")
    log_path = os.path.join(args.output, "model.log")

    cmd = [
        sys.executable,
        fit_script,
        "-i", negative_fold,
        "-i", positive_fold,
        "-m", model_path,
        "-l", log_path,
        "-d", args.device,
        "-s", str(args.seq_length),
        "-b", str(args.batch_size),
        "-M", str(args.epochs),
        "-e", str(args.early_stop),
        "-v", str(args.valid_prop),
        "-r", str(args.seed),
    ]
    subprocess.check_call(cmd)
    print(f"Saved mirDNN model to {model_path}")


if __name__ == "__main__":
    main()
