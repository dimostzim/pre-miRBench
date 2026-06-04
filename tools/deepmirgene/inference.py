#!/usr/bin/env python
import argparse
import os


def read_fasta_ids(path):
    ids = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
    return ids


def load_upstream_namespace(source_dir):
    upstream = os.path.join(source_dir, "inference", "deepMiRGene.py")
    with open(upstream) as handle:
        text = handle.read()
    prefix = text.split("## create directories for results and models", 1)[0]
    namespace = {"__name__": "deepmirgene_inference_defs"}
    exec(compile(prefix, upstream, "exec"), namespace)
    return namespace


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True, help="Input FASTA file")
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--model", help="Optional custom model weights (.hdf5)")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    deepmirgene_src = os.path.join(base_dir, "deepmirgene_src")
    os.makedirs(args.output, exist_ok=True)

    input_path = os.path.abspath(args.input)
    output_file = os.path.join(os.path.abspath(args.output), "predictions.txt")

    if args.model:
        model_path = os.path.abspath(args.model)
    else:
        model_path = os.path.join(deepmirgene_src, "inference", "model", "new_test.hdf5")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"deepMiRGene model not found: {model_path}")

    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    namespace = load_upstream_namespace(deepmirgene_src)
    model = namespace["mymodel"](400, 16, 20, 10, 400, 100)
    model.load_weights(model_path)

    encoded = namespace["import_data"](input_path)
    x_test = namespace["one_hot_wrap"](encoded, 400, 16)
    predictions = model.predict(x_test, verbose=0)
    record_ids = read_fasta_ids(input_path)
    if len(record_ids) != len(predictions):
        raise ValueError(
            f"deepMiRGene produced {len(predictions)} predictions for {len(record_ids)} input records"
        )

    with open(output_file, "w") as handle:
        handle.write("record_id\tscore\tpredicted_label\n")
        for record_id, prediction in zip(record_ids, predictions.tolist()):
            score = float(prediction[0])
            predicted_label = 1 if score >= float(prediction[1]) else 0
            handle.write(f"{record_id}\t{score:.8f}\t{predicted_label}\n")


if __name__ == "__main__":
    main()
