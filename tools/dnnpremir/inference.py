#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np
from keras.models import load_model


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
    input_path = os.path.abspath(args.input)
    if args.model:
        model_path = os.path.abspath(args.model)
    else:
        model_path = os.path.join(dnnpremir_src, "src", "CNN", "CNN_model.h5")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"dnnPreMiR model not found: {model_path}")

    sys.path.insert(0, dnnpremir_src)
    old_cwd = os.getcwd()
    os.chdir(dnnpremir_src)
    try:
        import isPreMiR

        os.makedirs("./temp", exist_ok=True)
        records = list(read_fasta(input_path))
        vectors = []
        for name, sequence in records:
            with open("./temp/temp_sequence.fa", "w") as handle:
                handle.write(f">{name}\n{sequence}\n")
            seq_struct = isPreMiR.second_struct_predict(sequence)
            vectors.append(isPreMiR.transform_seq_struct(seq_struct))
        if not records:
            raise ValueError(f"No FASTA records found in {input_path}")

        model = load_model(model_path)
        predictions = model.predict(np.array(vectors))
    finally:
        os.chdir(old_cwd)

    with open(output_file, "w") as handle:
        handle.write("record_id\tscore\tpredicted_label\n")
        for (record_id, _), prediction in zip(records, predictions.tolist()):
            score = float(prediction[0])
            predicted_label = 1 if score >= float(prediction[1]) else 0
            handle.write(f"{record_id}\t{score:.8f}\t{predicted_label}\n")


if __name__ == "__main__":
    main()
