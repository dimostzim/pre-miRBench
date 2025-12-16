#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
import shutil


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True, help="Input FASTA file")
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--model", default="fine_tuned_cnn",
                   choices=["fine_tuned_cnn", "base_cnn"],
                   help="Pre-trained model to use")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    deepmir_src = os.path.join(base_dir, "deepmir_src")

    os.makedirs(args.output, exist_ok=True)

    # Create absolute path for input
    input_path = os.path.abspath(args.input)

    # The original predictor.py script runs from its own directory
    # and creates output in user_data/{basename}/
    # We'll run it and then copy results to our output directory

    cmd = [
        sys.executable,
        "predictor.py",
        input_path
    ]

    # Temporarily modify the model file if not using default
    model_file = os.path.join(deepmir_src, "models", f"{args.model}.h5")
    predictor_file = os.path.join(deepmir_src, "predictor.py")

    if args.model != "fine_tuned_cnn":
        # Read the predictor.py file
        with open(predictor_file, 'r') as f:
            content = f.read()

        # Temporarily replace the model filename
        original_model = "fine_tuned_cnn.h5"
        new_model = f"{args.model}.h5"
        modified_content = content.replace(
            f'MODEL_FILENAME = CURRENT_DIR + "/models/{original_model}"',
            f'MODEL_FILENAME = CURRENT_DIR + "/models/{new_model}"'
        )

        # Write back
        with open(predictor_file, 'w') as f:
            f.write(modified_content)

        restore_needed = True
    else:
        restore_needed = False

    try:
        # Run from deepmir_src directory so relative paths work
        # Ensure the conda env bin (where java lives) is on PATH
        env = os.environ.copy()
        env_bin = os.path.join(sys.prefix, "bin")
        env["PATH"] = env_bin + os.pathsep + env.get("PATH", "")
        env.setdefault("JAVA_HOME", sys.prefix)

        subprocess.check_call(cmd, cwd=deepmir_src, env=env)

        # Copy results to output directory
        input_basename = os.path.basename(input_path).split('.')[0]
        user_data_dir = os.path.join(deepmir_src, "user_data", input_basename)

        # Copy all outputs to our output directory
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

    finally:
        # Restore original predictor.py if we modified it
        if restore_needed:
            with open(predictor_file, 'w') as f:
                f.write(content)


if __name__ == "__main__":
    main()
