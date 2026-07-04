#!/usr/bin/env python3
import argparse
from collections import Counter
import re


GENERIC_ALIAS_VALUES = {
    "",
    "=",
    "<>",
    "assembled-molecule",
    "Chromosome",
    "Linkage Group",
    "Mitochondrion",
    "na",
    "Primary Assembly",
    "unlocalized-scaffold",
    "unplaced-scaffold",
}


def strip_version(name):
    return re.sub(r"\.\d+$", "", name)


def clean_alias(value):
    return value.strip().strip(",;")


def fasta_headers(path):
    headers = set()
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                headers.add(line[1:].strip().split()[0])
    return headers


def add_alias(aliases, alias, genome_name):
    alias = clean_alias(alias)
    if alias in GENERIC_ALIAS_VALUES:
        return
    aliases.setdefault(alias, genome_name)
    unversioned = strip_version(alias)
    if unversioned != alias:
        aliases.setdefault(unversioned, genome_name)


def genome_lookup(genome_names):
    lookup = {}
    for name in genome_names:
        add_alias(lookup, name, name)
    return lookup


def fasta_header_aliases(path, genome_names):
    aliases = {}
    with open(path) as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            primary = header.split()[0]
            if primary not in genome_names:
                continue
            add_alias(aliases, primary, primary)
            primary_base = strip_version(primary)

            for pattern in (
                r"\bscaffold_\d+\b",
                r"\bdd_Smes_g4_\d+\b",
                r"\bspur5_(?:contig|scaffold)_\d+\b",
                r"\bscf\d+\b",
                r"\bscaffold\d+\b",
            ):
                for match in re.finditer(pattern, header):
                    alias = match.group(0)
                    add_alias(aliases, alias, primary)
                    if alias.startswith("dd_Smes_g4_"):
                        add_alias(aliases, f"{primary_base}_{alias}", primary)

            for match in re.finditer(r"\bchromosome\s+([A-Za-z0-9_.-]+)", header):
                alias = clean_alias(match.group(1))
                add_alias(aliases, alias, primary)
                add_alias(aliases, f"chr{alias}", primary)
                add_alias(aliases, f"Chr{alias}", primary)
    return aliases


def load_aliases(path, genome_names):
    aliases = {}
    if not path:
        return aliases
    lookup = genome_lookup(genome_names)
    with open(path) as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.startswith("#"):
                continue
            parts = [clean_alias(part) for part in raw_line.rstrip("\n").split("\t")]
            parts = [part for part in parts if part not in GENERIC_ALIAS_VALUES]
            if len(parts) < 2:
                continue
            genome_name = None
            for part in parts:
                if part in lookup:
                    genome_name = lookup[part]
                    break
            if genome_name is None:
                continue
            for part in parts:
                add_alias(aliases, part, genome_name)
    return aliases


def candidate_names(chrom):
    names = [chrom]
    unversioned = strip_version(chrom)
    if unversioned != chrom:
        names.append(unversioned)
    if chrom.startswith("chr"):
        names.append(chrom[3:])
    else:
        names.append(f"chr{chrom}")
        names.append(f"Chr{chrom}")
    chr_match = re.match(r"^.+_Chr(.+)$", chrom)
    if chr_match:
        suffix = chr_match.group(1)
        names.extend([suffix, f"chr{suffix}", f"Chr{suffix}"])
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
    aliases = fasta_header_aliases(args.genome, genome_names)
    aliases.update(load_aliases(args.alias, genome_names))
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
