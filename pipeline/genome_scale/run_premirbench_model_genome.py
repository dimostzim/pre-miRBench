#!/usr/bin/env python3
"""
Score genome-scale candidates with one pre-miRBench model bundle, in parallel chunks.

The genome dataset directory comes from prepare_genome_candidates.py and holds
premirbench_inputs/genome_<species>/{input/samples.tsv,labels.csv}. Candidates are split into
chunks under <output-dir>/inputs/, each chunk is scored on CPU by a separate inference.py process
of the model package (pre-miRBench_model_v2 unless --model-dir and --tool name another one, such as
pre-miRBench_model_v3) pinned to --cores-per-chunk cores, and the chunk
predictions are concatenated into <output-dir>/predictions.csv with the columns written by
run_genome_inference.py. Completed chunks are reused on rerun; the output of an interrupted chunk
is moved to <chunk>_INCOMPLETE_<UTC time> and the chunk is scored again.
"""
import argparse
import csv
import queue
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FIELDS = ["tool", "split", "record_id", "species", "label", "score", "predicted_label", "input_order"]


def utc_stamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def write_chunks(samples, inputs, split, chunk_size):
    partial = inputs.with_name(inputs.name + ".partial")
    if partial.exists():
        partial.rename(partial.with_name(f"{inputs.name}_INCOMPLETE_{utc_stamp()}"))
    with open(samples) as handle:
        header = handle.readline()
        rows = handle.readlines()
    for index, start in enumerate(range(0, len(rows), chunk_size)):
        path = partial / f"{split}_c{index:03d}" / "input" / "samples.tsv"
        path.parent.mkdir(parents=True)
        path.write_text(header + "".join(rows[start:start + chunk_size]))
    partial.rename(inputs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--genome-dataset-dir", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--artifacts-dir", required=True, help="Model bundle with deployment_manifest.json.")
    parser.add_argument("--python", required=True, help="Python of the model environment.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--cores-per-chunk", type=int, default=4)
    parser.add_argument("--first-core", type=int, default=0, help="Chunk processes are pinned to consecutive cores from here.")
    parser.add_argument("--model-dir", help="Model package with inference.py (default: <repo>/pre-miRBench_model_v2).")
    parser.add_argument("--tool", default="premirbench_model_v2", help="Tool name written to predictions.csv.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir or f"{args.repo}/pre-miRBench_model_v2"
    split = f"genome_{args.species}"
    model_inputs = Path(args.genome_dataset_dir) / "premirbench_inputs" / split
    out = Path(args.output_dir)
    if (out / "predictions.csv").exists():
        raise SystemExit(f"{out / 'predictions.csv'} exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    inputs = out / "inputs"
    if not inputs.is_dir():
        write_chunks(model_inputs / "input" / "samples.tsv", inputs, split, args.chunk_size)
    names = sorted(p.name for p in inputs.glob(f"{split}_c*"))

    slots = queue.Queue()
    for slot in range(args.parallel):
        first = args.first_core + slot * args.cores_per_chunk
        slots.put(f"{first}-{first + args.cores_per_chunk - 1}")

    def run(name):
        chunk_out = out / name
        if (chunk_out / "predictions.csv").is_file():
            return name, 0
        stamp = utc_stamp()
        for path in (chunk_out, out / f"{name}.log"):
            if path.exists():
                path.rename(path.with_name(f"{path.stem}_INCOMPLETE_{stamp}{path.suffix}"))
        cores = slots.get()
        try:
            result = subprocess.run(
                ["taskset", "-c", cores, args.python, "-u", f"{model_dir}/inference.py",
                 "--input", str(inputs / name / "input"), "--output", str(chunk_out / "predictions.csv"),
                 "--artifacts-dir", args.artifacts_dir, "--device", "cpu"],
                stdout=open(out / f"{name}.log", "w"), stderr=subprocess.STDOUT,
            )
        finally:
            slots.put(cores)
        return name, result.returncode

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        results = list(executor.map(run, names))
    failed = [name for name, code in results if code != 0 or not (out / name / "predictions.csv").is_file()]
    if failed:
        raise SystemExit(f"{len(failed)} chunks failed, e.g. {failed[:3]}; see {out}/<chunk>.log")

    labels = {row["id"]: row["label"] for row in csv.DictReader(open(model_inputs / "labels.csv"))}
    partial = out / "predictions.csv.partial"
    order = 0
    with open(partial, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for name in names:
            chunk_ids = [row["id"] for row in csv.DictReader(open(inputs / name / "input" / "samples.tsv"), delimiter="\t")]
            predictions = list(csv.DictReader(open(out / name / "predictions.csv")))
            if [row["id"] for row in predictions] != chunk_ids:
                raise SystemExit(f"{name}: prediction IDs differ from the chunk input")
            for row in predictions:
                writer.writerow([args.tool, split, row["id"], args.species, labels[row["id"]], row["probability_1"], row["prediction"], order])
                order += 1
    if order != len(labels):
        raise SystemExit(f"{order} predictions for {len(labels)} candidates")
    partial.rename(out / "predictions.csv")
    print(f"{args.tool} {args.species}: {len(names)} chunks, {order} candidates -> {out / 'predictions.csv'}")


if __name__ == "__main__":
    main()
