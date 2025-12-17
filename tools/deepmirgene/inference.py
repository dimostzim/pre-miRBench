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
    p.add_argument("--model", help="Optional custom model weights (.hdf5)")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    deepmirgene_src = os.path.join(base_dir, "deepmirgene_src")
    inference_dir = os.path.join(deepmirgene_src, "inference")
    os.makedirs(args.output, exist_ok=True)

    input_path = os.path.abspath(args.input)
    output_file = os.path.join(os.path.abspath(args.output), "predictions.txt")

    target_model_path = os.path.join(inference_dir, "model", "new_test.hdf5")
    restore_backup = None

    if args.model:
        provided_model = os.path.abspath(args.model)
        # deepMiRGene script expects ./model/new_test.hdf5, so stage the requested file there.
        os.makedirs(os.path.dirname(target_model_path), exist_ok=True)

        if os.path.exists(target_model_path):
            restore_backup = target_model_path + ".bak"
            shutil.copy2(target_model_path, restore_backup)

        shutil.copy2(provided_model, target_model_path)

    cmd = [
        sys.executable,
        "deepMiRGene.py",
        "-i",
        input_path,
        "-o",
        output_file,
    ]

    env = os.environ.copy()
    env.setdefault("KERAS_BACKEND", "theano")
    env.setdefault("THEANO_FLAGS", "optimizer=None")

    try:
        subprocess.check_call(cmd, cwd=inference_dir, env=env)
    finally:
        if restore_backup and os.path.exists(restore_backup):
            shutil.move(restore_backup, target_model_path)


if __name__ == "__main__":
    main()
