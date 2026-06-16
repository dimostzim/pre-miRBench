#!/usr/bin/env python3
"""
Prepare tool-specific training inputs from one canonical training dataset.

The canonical dataset uses 0-based, half-open genomic coordinates and a common
sequence window. Tool adapters crop only when the model architecture requires a
shorter native sequence.
"""
import argparse
import csv
import os
from pathlib import Path


TOOLS = ("deepmir", "deepmirgene", "dnnpremir", "mirdnn", "mire2e", "mustard")

DEFAULT_TARGET_LENGTHS = {
    "deepmir": None,
    "deepmirgene": None,
    "dnnpremir": 180,
    "mirdnn": 160,
    "mire2e": 100,
    "mustard": None,
}

METADATA_FIELDS = [
    "record_id",
    "species",
    "split",
    "split_reason",
    "label",
    "window_id",
    "chrom",
    "source_start",
    "source_end",
    "prepared_start",
    "prepared_end",
    "strand",
    "source_length",
    "prepared_length",
    "mirna_id",
    "family_id",
    "precursor_sequence",
    "target_start",
    "target_end",
    "left_trim",
    "right_trim",
]


def normalize_sequence(sequence):
    return sequence.strip().upper().replace("T", "U")


def label_name(value):
    return "positive" if str(value) in {"1", "positive", "true", "True"} else "negative"


def split_name(value):
    return value or "train"


def parse_int(value):
    if value in (None, ""):
        return None
    return int(float(value))


def crop_bounds(row, sequence_length, target_length):
    if target_length is None or sequence_length <= target_length:
        return 0, sequence_length

    max_left = sequence_length - target_length
    source_start = parse_int(row["start"])
    target_start = parse_int(row.get("target_start"))
    target_end = parse_int(row.get("target_end"))

    if target_start is None or target_end is None or source_start is None:
        left = max_left // 2
        return left, left + target_length

    rel_start = target_start - source_start
    rel_end = target_end - source_start
    target_center_left = int(round(((rel_start + rel_end) / 2.0) - (target_length / 2.0)))

    min_left = max(0, rel_end - target_length)
    max_target_left = min(max_left, rel_start)
    if min_left <= max_target_left:
        left = min(max(target_center_left, min_left), max_target_left)
    else:
        left = min(max(target_center_left, 0), max_left)
    return left, left + target_length


def prepare_row(row, tool, target_length):
    sequence = normalize_sequence(row["sequence"])
    left, right = crop_bounds(row, len(sequence), target_length)
    source_start = int(row["start"])
    source_end = int(row["end"])
    prepared_start = source_start + left
    prepared_end = source_start + right
    if row.get("strand") == "-":
        prepared_sequence = sequence[len(sequence) - right:len(sequence) - left]
    else:
        prepared_sequence = sequence[left:right]

    prepared = {
        "record_id": row["record_id"],
        "species": row.get("species", ""),
        "split": split_name(row.get("split")),
        "split_reason": row.get("split_reason", ""),
        "label": label_name(row["label"]),
        "window_id": row["window_id"],
        "chrom": row["chrom"],
        "source_start": source_start,
        "source_end": source_end,
        "prepared_start": prepared_start,
        "prepared_end": prepared_end,
        "strand": row.get("strand") or "+",
        "source_length": len(sequence),
        "prepared_length": len(prepared_sequence),
        "mirna_id": row.get("mirna_id", ""),
        "family_id": row.get("family_id", ""),
        "precursor_sequence": row.get("precursor_sequence", ""),
        "target_start": row.get("target_start", ""),
        "target_end": row.get("target_end", ""),
        "left_trim": left,
        "right_trim": len(sequence) - right,
        "sequence": prepared_sequence,
        "tool": tool,
    }
    return prepared


def write_fasta(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(f">{row['record_id']}\n")
            sequence = row["sequence"]
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset:offset + 80] + "\n")


def write_bed(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            score = "1" if row["label"] == "positive" else "0"
            handle.write(
                "\t".join(
                    [
                        row["chrom"],
                        str(row["prepared_start"]),
                        str(row["prepared_end"]),
                        row["record_id"],
                        score,
                        row["strand"],
                    ]
                )
                + "\n"
            )


def write_metadata(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METADATA_FIELDS})


def write_split_files(output_dir, rows):
    split_prefixes = {
        "train": "",
        "valid": "validation_",
        "test_known_species_known_family": "test_known_species_known_family_",
        "test_known_species_heldout_family": "test_known_species_heldout_family_",
        "test_heldout_species_known_family": "test_heldout_species_known_family_",
        "test_heldout_species_heldout_family": "test_heldout_species_heldout_family_",
    }
    for split, prefix in split_prefixes.items():
        split_rows = [row for row in rows if row["split"] == split]
        for label in ("positive", "negative"):
            label_rows = [row for row in split_rows if row["label"] == label]
            stem = "positive" if label == "positive" else "negative"
            write_fasta(output_dir / f"{prefix}{stem}.fa", label_rows)
            write_bed(output_dir / f"{prefix}{stem}.bed", label_rows)


def prepare_tool_inputs(rows, output_dir, target_lengths=None, tools=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_lengths = target_lengths or DEFAULT_TARGET_LENGTHS
    tools = tools or TOOLS
    outputs = {}

    for tool in tools:
        tool_dir = output_dir / tool
        tool_dir.mkdir(parents=True, exist_ok=True)
        prepared_rows = [prepare_row(row, tool, target_lengths.get(tool)) for row in rows]
        write_metadata(tool_dir / "metadata.csv", prepared_rows)
        write_split_files(tool_dir, prepared_rows)
        outputs[tool] = tool_dir
    return outputs


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def parse_tools(value):
    if not value or value == "all":
        return list(TOOLS)
    tools = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(tools) - set(TOOLS))
    if unknown:
        raise ValueError(f"Unsupported tools: {', '.join(unknown)}")
    return tools


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare tool-specific inputs from a canonical training dataset CSV.")
    parser.add_argument("--input", default="data/train/dataset.csv")
    parser.add_argument("--output-dir", default="data/train/tool_inputs")
    parser.add_argument("--tools", default="all")
    parser.add_argument("--dnnpremir-length", type=int, default=180)
    parser.add_argument("--mirdnn-length", type=int, default=160)
    parser.add_argument("--mire2e-length", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    target_lengths = DEFAULT_TARGET_LENGTHS.copy()
    target_lengths.update(
        {
            "dnnpremir": args.dnnpremir_length,
            "mirdnn": args.mirdnn_length,
            "mire2e": args.mire2e_length,
        }
    )
    outputs = prepare_tool_inputs(rows, args.output_dir, target_lengths=target_lengths, tools=parse_tools(args.tools))
    for tool, path in outputs.items():
        print(f"{tool}: {path}")


if __name__ == "__main__":
    main()
