#!/usr/bin/env python
import argparse
import gzip
import glob
import os
import random
import shutil
import subprocess


def require_tensorflow_gpu(device):
    if str(device).lower() == "cpu":
        raise SystemExit("MuStARD training requires a CUDA GPU; got --device cpu")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("MuStARD training requires a visible TensorFlow GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def class_label_map(class_list):
    classes = [int(value) for value in class_list.split(",")]
    if len(classes) != 2:
        raise ValueError("Explicit MuStARD split training currently expects exactly two classes")

    unique_classes = sorted(set(classes))
    one_hot_classes = []
    for i in range(len(unique_classes)):
        one_hot_classes.append(["1" if i == j else "0" for j in range(len(unique_classes))])

    # Match upstream MuStARD's fixed-label mapping, including classList=0,1.
    return {
        class_number: "_".join(one_hot_classes[class_number - 1])
        for class_number in unique_classes
    }


def write_reinforced_bed(output_path, inputs, labels, max_size, ext_flag, reinf_num):
    with gzip.open(output_path, "wt") as output:
        for input_path, class_number in inputs:
            label = labels[class_number]
            with open(input_path) as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    fields = line.split("\t")
                    if len(fields) < 6:
                        raise ValueError(f"Expected six-column BED line in {input_path}: {line}")
                    chrom, start, stop, name, _, strand = fields[:6]
                    start = int(start)
                    stop = int(stop)
                    if stop - start > max_size:
                        continue
                    for idx in range(1, reinf_num + 1):
                        new_start = start
                        new_stop = stop
                        if ext_flag == 1:
                            missing_space = max_size - (stop - start)
                            random_spot = random.randrange(missing_space) if missing_space > 0 else 0
                            new_start = start - random_spot
                            new_stop = stop + (missing_space - random_spot)
                        output.write(
                            "\t".join(
                                [
                                    chrom,
                                    str(new_start),
                                    str(new_stop),
                                    f"{name}_reinforced{idx}",
                                    label,
                                    strand,
                                ]
                            )
                            + "\n"
                        )


def run_explicit_split_training(args, base_dir):
    split_paths = {
        "test": (args.testPositiveIntervals, args.testNegativeIntervals),
        "validation": (args.validationPositiveIntervals, args.validationNegativeIntervals),
    }
    missing = [
        name
        for values in split_paths.values()
        for name, value in zip(("positive", "negative"), values)
        if value is None
    ]
    if missing:
        raise ValueError("Explicit MuStARD split training requires test and validation positive/negative BED files")

    labels = class_label_map(args.classList)
    classes = [int(value) for value in args.classList.split(",")]
    data_dir = os.path.join(args.output, "Data")
    model_dir = os.path.join(args.output, "Models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    write_reinforced_bed(
        os.path.join(data_dir, "train.bed.gz"),
        [(os.path.abspath(args.positiveIntervals), classes[0]), (os.path.abspath(args.negativeIntervals), classes[1])],
        labels,
        args.maxSize,
        args.extFlag,
        args.reinfNum,
    )
    write_reinforced_bed(
        os.path.join(data_dir, "test.bed.gz"),
        [(os.path.abspath(args.testPositiveIntervals), classes[0]), (os.path.abspath(args.testNegativeIntervals), classes[1])],
        labels,
        args.maxSize,
        args.extFlag,
        args.reinfNum,
    )
    write_reinforced_bed(
        os.path.join(data_dir, "validation.bed.gz"),
        [
            (os.path.abspath(args.validationPositiveIntervals), classes[0]),
            (os.path.abspath(args.validationNegativeIntervals), classes[1]),
        ],
        labels,
        args.maxSize,
        args.extFlag,
        args.reinfNum,
    )

    perl_code = r"""
use Files::CleanUp;
use Models::DNN;
Files::CleanUp::finalize_sequence_files($ARGV[0], "test,train,validation", $ARGV[1], $ARGV[2], $ARGV[3], $ARGV[4], 2);
Files::CleanUp::finalize_files($ARGV[0], "test,train,validation", $ARGV[5], $ARGV[6], $ARGV[7]) unless $ARGV[5] eq "sequence";
Models::DNN::train_DNN($ARGV[0], $ARGV[8], $ARGV[9], $ARGV[5], $ARGV[10]);
"""
    subprocess.check_call(
        [
            "perl",
            "-I",
            os.path.join(base_dir, "mustard_src", "src", "lib", "perl"),
            "-e",
            perl_code,
            data_dir,
            os.path.abspath(args.genome),
            str(args.maxSize),
            str(args.extFlag),
            str(args.shufClassFlag),
            args.inputMode,
            os.path.abspath(args.consDir),
            str(args.threads),
            model_dir,
            os.path.join(base_dir, "mustard_src", "src", "utilities", "python"),
            args.modelType,
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Train a MuStARD model.")
    parser.add_argument("--positiveIntervals", required=True)
    parser.add_argument("--negativeIntervals", required=True)
    parser.add_argument("--testPositiveIntervals")
    parser.add_argument("--testNegativeIntervals")
    parser.add_argument("--validationPositiveIntervals")
    parser.add_argument("--validationNegativeIntervals")
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

    explicit_split_args = (
        args.testPositiveIntervals,
        args.testNegativeIntervals,
        args.validationPositiveIntervals,
        args.validationNegativeIntervals,
    )
    if any(explicit_split_args):
        run_explicit_split_training(args, base_dir)
    else:
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
