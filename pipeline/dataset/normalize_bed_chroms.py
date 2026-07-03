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


def load_aliases(path, genome_names):
    aliases = {}
    if not path:
        return aliases
    with open(path) as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.startswith("#"):
                continue
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            alias, chrom = parts[0], parts[1]
            if chrom in genome_names:
                aliases[alias] = chrom
    return aliases


def candidate_names(chrom):
    names = [chrom]
    if chrom.startswith("chr"):
        names.append(chrom[3:])
    else:
        names.append(f"chr{chrom}")
    return names


def mapped_chrom(chrom, genome_names, aliases):
    if chrom in genome_names:
        return chrom
    if chrom in aliases:
        return aliases[chrom]
    for candidate in candidate_names(chrom):
        if candidate in genome_names:
            return candidate
        if candidate in aliases:
            return aliases[candidate]
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize BED contig names to match a genome FASTA.")
    parser.add_argument("--bed", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--alias", default=None, help="Optional UCSC chromAlias.txt file.")
    parser.add_argument("--min-matched-rows", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    genome_names = fasta_headers(args.genome)
    aliases = load_aliases(args.alias, genome_names)
    total_rows = 0
    matched_rows = 0
    mappings = Counter()
    dropped = Counter()

    with open(args.bed) as input_handle, open(args.output, "w") as output_handle:
        for raw_line in input_handle:
            if not raw_line.strip() or raw_line.startswith("#"):
                output_handle.write(raw_line)
                continue
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            total_rows += 1
            original = parts[0]
            normalized = mapped_chrom(original, genome_names, aliases)
            if normalized is None:
                dropped[original] += 1
                continue
            parts[0] = normalized
            mappings[(original, normalized)] += 1
            matched_rows += 1
            output_handle.write("\t".join(parts) + "\n")

    with open(args.report, "w") as report:
        report.write(f"genome_sequences: {len(genome_names)}\n")
        report.write(f"aliases: {len(aliases)}\n")
        report.write(f"bed_rows: {total_rows}\n")
        report.write(f"matched_rows: {matched_rows}\n")
        report.write(f"dropped_rows: {total_rows - matched_rows}\n")
        report.write("mappings:\n")
        for (original, normalized), count in sorted(mappings.items(), key=lambda item: (item[0][0], item[0][1])):
            report.write(f"  {original}\t{normalized}\t{count}\n")
        if dropped:
            report.write("dropped_chroms:\n")
            for chrom, count in sorted(dropped.items(), key=lambda item: (-item[1], item[0])):
                report.write(f"  {chrom}\t{count}\n")

    print(f"bed_rows: {total_rows}")
    print(f"matched_rows: {matched_rows}")
    print(f"dropped_rows: {total_rows - matched_rows}")
    print(f"output: {args.output}")
    print(f"report: {args.report}")

    if matched_rows < args.min_matched_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
