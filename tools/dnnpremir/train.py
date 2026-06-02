#!/usr/bin/env python
import argparse
import csv
import os
import random
import subprocess
import sys

import numpy as np


def require_tensorflow_gpu(device):
    if str(device).lower() == "cpu":
        raise SystemExit("dnnPreMiR training requires a CUDA GPU; got --device cpu")
    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        raise SystemExit("dnnPreMiR training requires a visible TensorFlow GPU")


def patch_keras_optimizers():
    import keras

    if not hasattr(keras.optimizers, "Adam") and hasattr(keras.optimizers, "adam_v2"):
        keras.optimizers.Adam = keras.optimizers.adam_v2.Adam


X_CAST = {
    "A.": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "U.": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "G.": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "C.": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "A(": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "U(": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "G(": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "C(": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "A)": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "U)": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "G)": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "C)": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}


def encode_seq_struct(value):
    items = value.strip().split()[:180]
    encoded = [X_CAST[item] for item in items]
    while len(encoded) < 180:
        encoded.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return encoded


def read_csv_examples(path, label):
    examples = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if "seq_struc" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a seq_struc column")
        y = [1, 0] if label else [0, 1]
        for row in reader:
            examples.append((encode_seq_struct(row["seq_struc"]), y))
    return examples


def read_fasta(path):
    name = None
    parts = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(parts)
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
    if name is not None:
        yield name, "".join(parts)


def fold_sequence(name, sequence):
    rnafold = os.path.join(os.path.dirname(sys.executable), "RNAfold")
    clean_sequence = sequence.upper().replace("T", "U")
    process = subprocess.run(
        [rnafold, "--noPS"],
        input=f">{name}\n{clean_sequence}\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"RNAfold produced unexpected output for {name}: {process.stdout}")
    return lines[1], lines[2].split()[0]


def seq_struct_value(sequence, structure):
    return " ".join(sequence[i] + structure[i] for i in range(min(len(sequence), len(structure))))


def fasta_to_csv(fasta_path, csv_path, label):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    count = 0
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Accession", "seq_struc", "Classification"])
        writer.writeheader()
        for name, sequence in read_fasta(fasta_path):
            folded_sequence, structure = fold_sequence(name, sequence)
            writer.writerow(
                {
                    "Accession": name,
                    "seq_struc": seq_struct_value(folded_sequence, structure),
                    "Classification": "TRUE" if label else "FALSE",
                }
            )
            count += 1
    if count == 0:
        raise ValueError(f"No sequences found in {fasta_path}")
    return csv_path


def load_model_factory(source_dir, architecture):
    patch_keras_optimizers()

    if architecture == "cnn":
        sys.path.insert(0, os.path.join(source_dir, "src", "CNN"))
        from CNNModel import CNN_model

        return CNN_model
    if architecture == "rnn":
        sys.path.insert(0, os.path.join(source_dir, "src", "RNN"))
        from RNNModel import RNN_model

        return RNN_model
    if architecture == "cnn_rnn":
        sys.path.insert(0, os.path.join(source_dir, "src", "CNN_RNN"))
        from CNNRNNModel import CNN_RNN_model

        return CNN_RNN_model
    raise ValueError(f"Unsupported architecture: {architecture}")


def main():
    parser = argparse.ArgumentParser(description="Train a dnnPreMiR model.")
    parser.add_argument("--positive_fasta")
    parser.add_argument("--negative_fasta")
    parser.add_argument("--validation_positive_fasta")
    parser.add_argument("--validation_negative_fasta")
    parser.add_argument("--positive_csv")
    parser.add_argument("--negative_csv")
    parser.add_argument("--validation_positive_csv")
    parser.add_argument("--validation_negative_csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", choices=["cnn", "rnn", "cnn_rnn"], default="cnn")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    require_tensorflow_gpu(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.positive_csv and args.negative_csv:
        positive_csv = args.positive_csv
        negative_csv = args.negative_csv
    elif args.positive_fasta and args.negative_fasta:
        preprocess_dir = os.path.join(args.output, "preprocessed")
        positive_csv = fasta_to_csv(args.positive_fasta, os.path.join(preprocess_dir, "positive.csv"), True)
        negative_csv = fasta_to_csv(args.negative_fasta, os.path.join(preprocess_dir, "negative.csv"), False)
    else:
        raise ValueError("Provide either positive_csv/negative_csv or positive_fasta/negative_fasta")

    validation_data = None
    validation_split = args.validation_split
    if args.validation_positive_csv and args.validation_negative_csv:
        validation_positive_csv = args.validation_positive_csv
        validation_negative_csv = args.validation_negative_csv
    elif args.validation_positive_fasta and args.validation_negative_fasta:
        preprocess_dir = os.path.join(args.output, "preprocessed")
        validation_positive_csv = fasta_to_csv(
            args.validation_positive_fasta,
            os.path.join(preprocess_dir, "validation_positive.csv"),
            True,
        )
        validation_negative_csv = fasta_to_csv(
            args.validation_negative_fasta,
            os.path.join(preprocess_dir, "validation_negative.csv"),
            False,
        )
    else:
        validation_positive_csv = None
        validation_negative_csv = None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, "dnnpremir_src")
    model_factory = load_model_factory(source_dir, args.architecture)

    examples = read_csv_examples(positive_csv, True)
    examples.extend(read_csv_examples(negative_csv, False))
    random.shuffle(examples)

    x_dataset = np.array([x for x, _ in examples])
    y_dataset = np.array([y for _, y in examples])

    if validation_positive_csv and validation_negative_csv:
        validation_examples = read_csv_examples(validation_positive_csv, True)
        validation_examples.extend(read_csv_examples(validation_negative_csv, False))
        validation_data = (
            np.array([x for x, _ in validation_examples]),
            np.array([y for _, y in validation_examples]),
        )
        validation_split = 0.0

    model = model_factory()
    model.fit(
        x_dataset,
        y_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=validation_split,
        validation_data=validation_data,
    )
    output_model = os.path.join(args.output, "CNN_model.h5")
    model.save(output_model)
    print(f"Saved dnnPreMiR model to {output_model}")


if __name__ == "__main__":
    main()
