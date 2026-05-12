#!/usr/bin/env python
import argparse
import os
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True, help="Input FASTA file")
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--model", help="Optional custom CNN_model.h5 path")
    p.add_argument("--seq_length", type=int, default=180, help="Sequence length (fixed at 180, for documentation only)")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dnnpremir_src = os.path.join(base_dir, "dnnpremir_src")

    os.makedirs(args.output, exist_ok=True)

    output_file = os.path.join(os.path.abspath(args.output), "predictions.txt")

    # the original script expects to be run from its own directory
    # because it uses relative paths like ./bin/RNAfold and src/CNN/CNN_model.h5
    cmd = [
        sys.executable,
        "isPreMiR.py",
        "-i", os.path.abspath(args.input),
        "-o", output_file
    ]

    target_model_path = os.path.join(dnnpremir_src, "src", "CNN", "CNN_model.h5")
    restore_backup = None
    if args.model:
        provided_model = os.path.abspath(args.model)
        if not os.path.isfile(provided_model):
            raise FileNotFoundError(f"dnnPreMiR model not found: {provided_model}")
        if os.path.exists(target_model_path):
            restore_backup = target_model_path + ".bak"
            shutil.copy2(target_model_path, restore_backup)
        shutil.copy2(provided_model, target_model_path)

    try:
        subprocess.check_call(cmd, cwd=dnnpremir_src)
    finally:
        if restore_backup and os.path.exists(restore_backup):
            shutil.move(restore_backup, target_model_path)


if __name__ == "__main__":
    main()
