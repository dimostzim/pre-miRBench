#!/usr/bin/env python
import argparse
import glob
import os
import shutil
import subprocess


def require_tensorflow_gpu(device):
    if str(device).lower() == "cpu":
        raise SystemExit("MuStARD training requires a CUDA GPU; got --device cpu")
    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        raise SystemExit("MuStARD training requires a visible TensorFlow GPU")


def main():
    parser = argparse.ArgumentParser(description="Train a MuStARD model.")
    parser.add_argument("--positiveIntervals", required=True)
    parser.add_argument("--negativeIntervals", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--consDir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classList", default="0,1")
    parser.add_argument("--maxSize", type=int, default=200)
    parser.add_argument("--extFlag", type=int, default=0)
    parser.add_argument("--reinfNum", type=int, default=5)
    parser.add_argument("--shufClassFlag", type=int, default=0)
    parser.add_argument("--inputMode", default="sequence,RNAfold,conservation")
    parser.add_argument("--modelType", default="CNN")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--exclTest", default="chr1,chr3")
    parser.add_argument("--exclValid", default="chr2,chr4")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    require_tensorflow_gpu(args.device)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(base_dir, "mustard_src", "src", "MuStARD_train.pl")

    cmd = [
        "perl",
        train_script,
        "--list", f"{os.path.abspath(args.positiveIntervals)},{os.path.abspath(args.negativeIntervals)}",
        "--class", args.classList,
        "--dir", os.path.abspath(args.output),
        "--genome", os.path.abspath(args.genome),
        "--consDir", os.path.abspath(args.consDir),
        "--maxSize", str(args.maxSize),
        "--extFlag", str(args.extFlag),
        "--reinfNum", str(args.reinfNum),
        "--shufClassFlag", str(args.shufClassFlag),
        "--inputMode", args.inputMode,
        "--modelType", args.modelType,
        "--threads", str(args.threads),
        "--exclTest", args.exclTest,
        "--exclValid", args.exclValid,
    ]
    subprocess.check_call(cmd)

    candidates = glob.glob(os.path.join(args.output, "Models", "**", "CNNonRaw.hdf5"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"MuStARD training completed but no CNNonRaw.hdf5 was found in {args.output}")
    canonical_model = os.path.join(args.output, "CNNonRaw.hdf5")
    shutil.copy2(candidates[0], canonical_model)
    print(f"Saved MuStARD model to {canonical_model}")


if __name__ == "__main__":
    main()
