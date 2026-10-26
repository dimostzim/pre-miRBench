#!/usr/bin/env python3
"""
Score genome-scale candidates with one trained published predictor, in parallel chunks.

The genome dataset directory comes from prepare_genome_candidates.py and holds
tool_inputs/<tool>/genome_<species>_{positive,negative}.{fa,bed}. Candidates are split into
chunks written as extra splits (genome_<species>_cNNN), each chunk is scored by a separate
pipeline/evaluate.py process (one container each), and the chunk predictions are
concatenated into <output-dir>/predictions.csv. Completed chunks are reused on rerun; the output of an
interrupted chunk is moved to <chunk>_INCOMPLETE_<UTC time> and the chunk is scored again.
"""
import argparse
import csv
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FASTA_TOOLS = {"deepmir", "deepmirgene", "dnnpremir", "mirdnn", "mire2e"}


def read_fasta(path):
    records, name, chunks = [], None, []
    for line in open(path):
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks)))
            name, chunks = line[1:].strip(), []
        else:
            chunks.append(line.strip())
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def read_lines(path):
    return [line for line in open(path) if line.strip()]


def write_chunks(tool_dir, split, chunk_size, fasta):
    suffix = "fa" if fasta else "bed"
    chunks = []
    for label in ("positive", "negative"):
        path = tool_dir / f"{split}_{label}.{suffix}"
        items = read_fasta(path) if fasta else read_lines(path)
        for index, start in enumerate(range(0, max(len(items), 1), chunk_size)):
            while len(chunks) <= index:
                chunks.append({"positive": [], "negative": []})
            chunks[index][label] = items[start:start + chunk_size]
    names = []
    for index, chunk in enumerate(chunks):
        name = f"{split}_c{index:03d}"
        for label in ("positive", "negative"):
            with open(tool_dir / f"{name}_{label}.{suffix}", "w") as handle:
                for item in chunk[label]:
                    handle.write(f">{item[0]}\n{item[1]}\n" if fasta else item)
        names.append(name)
    return names


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--genome-dataset-dir", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--cpu-only", action="store_true", help="Score without a GPU (passed to evaluate.py).")
    parser.add_argument("--cpu-threads", type=int, default=4, help="With --cpu-only, cores per chunk container (passed to evaluate.py).")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = Path(args.genome_dataset_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = f"genome_{args.species}"
    tool_dir = dataset / "tool_inputs" / args.tool
    names = sorted(p.name.rsplit("_", 1)[0] for p in tool_dir.glob(f"{split}_c*_positive.*"))
    if not names:
        names = write_chunks(tool_dir, split, args.chunk_size, args.tool in FASTA_TOOLS)

    def run(name):
        chunk_out = out / name
        if (chunk_out / "predictions.csv").is_file():
            return name, 0
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        for path in (chunk_out, out / f"{name}.log"):
            if path.exists():
                path.rename(path.with_name(f"{path.stem}_INCOMPLETE_{stamp}{path.suffix}"))
        result = subprocess.run(
            ["python3", "-u", f"{args.repo}/pipeline/evaluate.py", "--dataset-dir", str(dataset), "--training-root", args.training_root,
             "--run-name", args.run_name, "--tools", args.tool, "--splits", name, "--output-dir", str(chunk_out),
             *(["--cpu-only", "--cpu-threads", str(args.cpu_threads)] if args.cpu_only else [])],
            stdout=open(out / f"{name}.log", "w"), stderr=subprocess.STDOUT,
        )
        return name, result.returncode

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        results = list(executor.map(run, names))
    failed = [name for name, code in results if code != 0 or not (out / name / "predictions.csv").is_file()]
    if failed:
        raise SystemExit(f"{len(failed)} chunks failed, e.g. {failed[:3]}; see {out}/<chunk>.log")

    with open(out / "predictions.csv", "w", newline="") as handle:
        writer = None
        for name in names:
            for row in csv.DictReader(open(out / name / "predictions.csv")):
                row["split"] = split
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
    print(f"{args.tool} {args.species}: {len(names)} chunks -> {out / 'predictions.csv'}")


if __name__ == "__main__":
    main()
