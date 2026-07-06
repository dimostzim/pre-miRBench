#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
import shutil
import tempfile


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True, help="Input FASTA file")
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--model", default="fine_tuned_cnn",
                   help="Pre-trained model name or explicit .h5 model path")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    deepmir_src = os.path.join(base_dir, "deepmir_src")

    os.makedirs(args.output, exist_ok=True)

    # DeepMir uses pyfaidx, which writes a .fai index beside the FASTA. The
    # evaluator mounts inputs read-only, so run the predictor on a writable copy.
    input_path = os.path.abspath(args.input)
    temp_dir = tempfile.TemporaryDirectory()
    writable_input = os.path.join(temp_dir.name, os.path.basename(input_path))
    shutil.copy2(input_path, writable_input)

    # The original predictor.py script runs from its own directory
    # and creates output in user_data/{basename}/
    # We'll run it and then copy results to our output directory

    cmd = [
        sys.executable,
        "predictor.py",
        writable_input,
    ]

    if os.path.isfile(args.model):
        model_file = os.path.abspath(args.model)
    else:
        model_file = os.path.join(deepmir_src, "models", f"{args.model}.h5")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"DeepMir model not found: {model_file}")

    # Run from deepmir_src directory so relative paths work.
    env = os.environ.copy()
    env_bin = os.path.join(sys.prefix, "bin")
    env["PATH"] = env_bin + os.pathsep + env.get("PATH", "")
    env.setdefault("JAVA_HOME", sys.prefix)
    env["DEEPMIR_MODEL_FILENAME"] = model_file

    subprocess.check_call(cmd, cwd=deepmir_src, env=env)

    input_basename = os.path.basename(writable_input).split('.')[0]
    user_data_dir = os.path.join(deepmir_src, "user_data", input_basename)

    if os.path.exists(user_data_dir):
        for item in os.listdir(user_data_dir):
            src = os.path.join(user_data_dir, item)
            dst = os.path.join(args.output, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        print(f"Results copied to {args.output}/")
    else:
        print(f"Warning: Expected output directory {user_data_dir} not found")


if __name__ == "__main__":
    main()
