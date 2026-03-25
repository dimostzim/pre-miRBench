#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare FASTA/BED inputs for tool benchmarking from a benchmark dataset CSV."
    )
    p.add_argument("--input", required=True, help="Input benchmark dataset CSV")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write prepared inputs into",
    )
    p.add_argument(
        "--prefix",
        default=None,
        help="Optional output filename prefix (default: input CSV stem)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    prefix = args.prefix or input_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = output_dir / f"{prefix}.fa"
    bed_path = output_dir / f"{prefix}.bed"
    metadata_path = output_dir / f"{prefix}.metadata.csv"

    with input_path.open(newline="") as src, \
            fasta_path.open("w") as fasta_out, \
            bed_path.open("w", newline="") as bed_out, \
            metadata_path.open("w", newline="") as meta_out:
        reader = csv.DictReader(src)

        meta_fieldnames = [
            "window_id",
            "chrom",
            "start",
            "end",
            "strand",
            "label",
            "sequence_length",
        ]
        if reader.fieldnames and "target_mirna" in reader.fieldnames:
            meta_fieldnames.append("target_mirna")
        if reader.fieldnames and "contained_mirnas" in reader.fieldnames:
            meta_fieldnames.append("contained_mirnas")

        meta_writer = csv.DictWriter(meta_out, fieldnames=meta_fieldnames)
        meta_writer.writeheader()

        for row in reader:
            window_id = row["window_id"]
            sequence = row["sequence"].strip().upper().replace("T", "U")
            chrom = row["chrom"]
            start = row["start"]
            end = row["end"]
            strand = row.get("strand", ".")
            label = row.get("label", "")

            fasta_out.write(f">{window_id}\n{sequence}\n")
            bed_out.write(f"{chrom}\t{start}\t{end}\t{window_id}\t0\t{strand}\n")

            meta_row = {
                "window_id": window_id,
                "chrom": chrom,
                "start": start,
                "end": end,
                "strand": strand,
                "label": label,
                "sequence_length": len(sequence),
            }
            if "target_mirna" in row:
                meta_row["target_mirna"] = row["target_mirna"]
            if "contained_mirnas" in row:
                meta_row["contained_mirnas"] = row["contained_mirnas"]
            meta_writer.writerow(meta_row)

    print(f"FASTA: {fasta_path}")
    print(f"BED: {bed_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
