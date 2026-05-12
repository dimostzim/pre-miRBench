#!/usr/bin/env python
import argparse
import csv
import os
import random
import sys

import numpy as np


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


def load_model_factory(source_dir, architecture):
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
    parser.add_argument("--positive_csv", required=True)
    parser.add_argument("--negative_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", choices=["cnn", "rnn", "cnn_rnn"], default="cnn")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, "dnnpremir_src")
    model_factory = load_model_factory(source_dir, args.architecture)

    examples = read_csv_examples(args.positive_csv, True)
    examples.extend(read_csv_examples(args.negative_csv, False))
    random.shuffle(examples)

    x_dataset = np.array([x for x, _ in examples])
    y_dataset = np.array([y for _, y in examples])

    model = model_factory()
    model.fit(
        x_dataset,
        y_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
    )
    output_model = os.path.join(args.output, "CNN_model.h5")
    model.save(output_model)
    print(f"Saved dnnPreMiR model to {output_model}")


if __name__ == "__main__":
    main()
