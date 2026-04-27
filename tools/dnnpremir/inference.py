#!/usr/bin/env python
import argparse
import csv
import os
import subprocess
import sys


def _read_fasta_seqs(fasta_path):
    """Return {record_id: sequence} preserving input order."""
    seqs = {}
    current_id = None
    current_seq = []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(current_seq)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line.upper().replace("T", "U"))
    if current_id is not None:
        seqs[current_id] = "".join(current_seq)
    return seqs


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True, help="Input FASTA file")
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--seq_length", type=int, default=180,
                   help="Sequence length (fixed at 180, for documentation only)")
    p.add_argument(
        "--norm-output", dest="norm_output", default="y", choices=["y", "n"],
        help="y: unified predictions.csv; n: original predictions.txt block format",
    )
    args = p.parse_args()
    norm_output = args.norm_output.lower() == "y"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dnnpremir_src = os.path.join(base_dir, "dnnpremir_src")

    os.makedirs(args.output, exist_ok=True)

    # Upstream always writes CSV (patched at build time).
    csv_output_file = os.path.join(os.path.abspath(args.output), "predictions.csv")

    cmd = [
        sys.executable,
        "isPreMiR.py",
        "-i", os.path.abspath(args.input),
        "-o", csv_output_file,
    ]

    subprocess.check_call(cmd, cwd=dnnpremir_src)

    if not norm_output:
        # Original format: block text  >id\nSEQUENCE  True/False\n===
        SEP = "=" * 27
        seqs = _read_fasta_seqs(os.path.abspath(args.input))
        txt_output_file = os.path.join(os.path.abspath(args.output), "predictions.txt")
        with open(csv_output_file, newline="") as csv_fh, \
             open(txt_output_file, "w") as txt_fh:
            for row in csv.DictReader(csv_fh):
                rid = row["window_id"]
                label = "True" if float(row["probability_score"]) >= 0.5 else "False"
                seq = seqs.get(rid, "N" * 180)
                txt_fh.write(f">{rid}\n{seq}  {label}\n{SEP}\n")
        os.remove(csv_output_file)


if __name__ == "__main__":
    main()
