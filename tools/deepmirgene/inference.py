#!/usr/bin/env python
import argparse
import csv
import os
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True, help="Input FASTA file")
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--model", help="Optional custom model weights (.hdf5)")
    p.add_argument(
        "--norm-output", dest="norm_output", default="y", choices=["y", "n"],
        help="y: unified predictions.csv; n: original predictions.txt (one int per line)",
    )
    args = p.parse_args()
    norm_output = args.norm_output.lower() == "y"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    deepmirgene_src = os.path.join(base_dir, "deepmirgene_src")
    inference_dir = os.path.join(deepmirgene_src, "inference")
    os.makedirs(args.output, exist_ok=True)

    input_path = os.path.abspath(args.input)
    # Upstream (patched) always writes unified CSV; we may rewrite below.
    csv_output_file = os.path.join(os.path.abspath(args.output), "predictions.csv")

    target_model_path = os.path.join(inference_dir, "model", "new_test.hdf5")
    restore_backup = None

    if args.model:
        provided_model = os.path.abspath(args.model)
        os.makedirs(os.path.dirname(target_model_path), exist_ok=True)
        if os.path.exists(target_model_path):
            restore_backup = target_model_path + ".bak"
            shutil.copy2(target_model_path, restore_backup)
        shutil.copy2(provided_model, target_model_path)

    cmd = [
        sys.executable,
        "deepMiRGene.py",
        "-i", input_path,
        "-o", csv_output_file,
    ]

    env = os.environ.copy()
    env.setdefault("KERAS_BACKEND", "tensorflow")

    try:
        subprocess.check_call(cmd, cwd=inference_dir, env=env)
    finally:
        if restore_backup and os.path.exists(restore_backup):
            shutil.move(restore_backup, target_model_path)

    if not norm_output:
        # Original format: predictions.txt — one integer per line (1=pre-miRNA, 0=other)
        txt_output_file = os.path.join(os.path.abspath(args.output), "predictions.txt")
        with open(csv_output_file, newline="") as csv_fh, \
             open(txt_output_file, "w") as txt_fh:
            for row in csv.DictReader(csv_fh):
                txt_fh.write("1\n" if float(row["probability_score"]) >= 0.5 else "0\n")
        os.remove(csv_output_file)


if __name__ == "__main__":
    main()
