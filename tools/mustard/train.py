#!/usr/bin/env python
import argparse
import gzip
import glob
import os
import random
import shutil
import subprocess
import sys


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


def run_checked(cmd):
    """Run a MuStARD training step, streaming its output. MuStARD's Perl code starts the Python
    trainer without checking its exit status, so a Python traceback in the output also counts as
    a failure (otherwise the last checkpoint of a crashed run would be saved as the model)."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    traceback_seen = False
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        traceback_seen = traceback_seen or line.startswith("Traceback (most recent call last)")
    if process.wait() != 0 or traceback_seen:
        raise SystemExit(f"MuStARD training failed: exit code {process.returncode}, Python traceback in output: {traceback_seen}")


UPSTREAM_SEED_LINES = ("np.random.seed(20)", "rn.seed(1984)", "tf.random.set_seed(8)")


def seed_training_script(script_text, seed):
    """Replace the three seeds hard-coded in MuStARD's train_CNN.py (numpy 20, random 1984, TensorFlow 8) by one seed."""
    for line, call in zip(UPSTREAM_SEED_LINES, ("np.random.seed", "rn.seed", "tf.random.set_seed")):
        if script_text.count(line) != 1:
            raise RuntimeError(f"Expected one '{line}' in train_CNN.py, found {script_text.count(line)}")
        script_text = script_text.replace(line, f"{call}({seed})")
    return script_text


# model.predict on the whole validation set leaks GPU memory in TF 2.6 at every epoch end, which ran the 16 GB GPU out of
# memory near epoch 275 (v3, 23 Oct 2026); predict_on_batch over slices of 1024 gives the same scores without the leak.
VALIDATION_PREDICT = "\t\tpredictions = self.model.predict(self.validation_x, verbose=0)\n"
BATCHED_VALIDATION_PREDICT = (
    "\t\tinputs = self.validation_x if isinstance(self.validation_x, list) else [self.validation_x]\n"
    "\t\tpredictions = np.concatenate([self.model.predict_on_batch([x[start:start + 1024] for x in inputs] if len(inputs) > 1 else inputs[0][start:start + 1024]) for start in range(0, len(inputs[0]), 1024)])\n"
)


def batch_validation_predict(script_text):
    """Make the validation-AUPRC callback added by patch_mustard_auprc.py predict in batches."""
    if script_text.count(VALIDATION_PREDICT) != 1:
        raise RuntimeError(f"Expected one validation predict line in train_CNN.py, found {script_text.count(VALIDATION_PREDICT)}")
    return script_text.replace(VALIDATION_PREDICT, BATCHED_VALIDATION_PREDICT)


def seeded_source(base_dir, output_dir, seed):
    """Copy the MuStARD source into the run folder with train_CNN.py seeded and its validation predict batched; return the copy's root.

    The whole tree is copied because train_CNN.py imports its library from <script dir>/../../lib/python.
    """
    target = os.path.join(output_dir, "seeded_mustard_src")
    shutil.copytree(os.path.join(base_dir, "mustard_src"), target)
    script = os.path.join(target, "src", "utilities", "python", "train_CNN.py")
    with open(script) as handle:
        text = batch_validation_predict(seed_training_script(handle.read(), seed))
    with open(script, "w") as handle:
        handle.write(text)
    return target


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
    source_root = seeded_source(base_dir, args.output, args.seed) if args.seed is not None else os.path.join(base_dir, "mustard_src")
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
    run_checked(
        [
            "perl",
            "-I",
            os.path.join(source_root, "src", "lib", "perl"),
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
            os.path.join(source_root, "src", "utilities", "python"),
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
    parser.add_argument("--seed", type=int, default=None, help="Replace MuStARD's hard-coded training seeds by this seed.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    # MuStARD skips training when a model already exists in the output folder, and the wrapper would then save that
    # earlier (possibly partial) model; refuse instead.
    existing = glob.glob(os.path.join(args.output, "Models", "**", "*.hdf5"), recursive=True)
    if existing or os.path.exists(os.path.join(args.output, "seeded_mustard_src")):
        raise SystemExit(f"{args.output} already holds output of an earlier run ({existing[:1] or 'seeded_mustard_src'}); use a new folder")
    require_tensorflow_gpu(args.device)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(base_dir, "mustard_src", "src", "MuStARD_train.pl")

    explicit_split_args = (
        args.testPositiveIntervals,
        args.testNegativeIntervals,
        args.validationPositiveIntervals,
        args.validationNegativeIntervals,
    )
    if args.seed is not None and not any(explicit_split_args):
        raise SystemExit("--seed is supported only with explicit test/validation interval files")
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
        run_checked(cmd)

    candidates = glob.glob(os.path.join(args.output, "Models", "**", "CNNonRaw.hdf5"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"MuStARD training completed but no CNNonRaw.hdf5 was found in {args.output}")
    canonical_model = os.path.join(args.output, "CNNonRaw.hdf5")
    shutil.copy2(candidates[0], canonical_model)
    print(f"Saved MuStARD model to {canonical_model}")


if __name__ == "__main__":
    main()
