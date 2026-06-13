#!/usr/bin/env python3
import argparse
from collections import Counter


def fasta_headers(path):
    headers = set()
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                headers.add(line[1:].strip().split()[0])
    return headers


def bed_chrom_counts(path):
    counts = Counter()
    with open(path) as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.startswith("#"):
                continue
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            counts[parts[0]] += 1
    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="Validate that BED chromosome/scaffold names exist in a genome FASTA.")
    parser.add_argument("--bed", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--allow-missing", action="store_true", help="Report missing names but exit successfully.")
    return parser.parse_args()


def main():
    args = parse_args()
    genome_names = fasta_headers(args.genome)
    bed_counts = bed_chrom_counts(args.bed)
    missing = {chrom: count for chrom, count in bed_counts.items() if chrom not in genome_names}
    matched_rows = sum(count for chrom, count in bed_counts.items() if chrom in genome_names)
    total_rows = sum(bed_counts.values())

    print(f"genome_sequences: {len(genome_names)}")
    print(f"bed_chroms: {len(bed_counts)}")
    print(f"bed_rows: {total_rows}")
    print(f"matched_rows: {matched_rows}")
    print(f"missing_rows: {total_rows - matched_rows}")

    if missing:
        print("missing_chroms:")
        for chrom, count in sorted(missing.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {chrom}\t{count}")
        if not args.allow_missing:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
