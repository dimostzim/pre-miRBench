#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from tool_adapters import (
    build_source_records,
    load_target_mirna_coords,
    parse_tools,
    write_source_manifest,
    write_tool_inputs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare collapsed 1:1 benchmark inputs for each tool using tool-specific sequence/interval lengths."
    )
    parser.add_argument("--input", default="benchmark/datasets/1_1_collapsed.csv", help="Saved collapsed 1:1 source dataset CSV")
    parser.add_argument("--output-dir", default="benchmark/prepared_inputs/1_1_collapsed", help="Directory to write prepared tool inputs into")
    parser.add_argument("--prefix", default="1_1_collapsed", help="Output prefix")
    parser.add_argument("--tools", default="all", help="Comma-separated tool list or 'all'")
    parser.add_argument("--truth-bed", default="benchmark/data/hsa-precursors-no-v2.bed", help="BED file with target pre-miRNA coordinates")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    prefix = args.prefix
    tools = parse_tools(args.tools)
    target_mirna_coords = load_target_mirna_coords(args.truth_bed)

    with input_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_records = build_source_records(rows, prefix, target_mirna_coords=target_mirna_coords)
    manifest_path = write_source_manifest(source_records, output_dir, prefix)
    print(f"Source manifest: {manifest_path}")

    for tool in tools:
        paths = write_tool_inputs(tool, source_records, output_dir / tool, prefix)
        print(f"{tool} FASTA: {paths['fasta']}")
        print(f"{tool} BED: {paths['bed']}")
        print(f"{tool} metadata: {paths['metadata']}")


if __name__ == "__main__":
    main()
