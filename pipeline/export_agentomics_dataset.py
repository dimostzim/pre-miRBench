#!/usr/bin/env python3
"""
Export a pre-miRBench dataset.csv in the Agentomics dataset layout.

train and validation are the agent-visible splits. The four test partitions are
written as directories whose names start with `test`, which Agentomics withholds
from the agent and uses only for final evaluation. Every split contains
labels.csv (id, label) and input/samples.tsv with the five model inputs used in
v1: id, species, sequence_rna, structure, mfe. Record IDs, coordinates, family
IDs, mining scores and split reasons are not exported because they encode label
provenance; sample IDs are neutral and split-local. Rows are shuffled within
each split before IDs are assigned, because dataset.csv groups rows by species
and label and sequential IDs would otherwise reveal the label.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

EXPORT_SPLITS = {
    "train": "train",
    "valid": "validation",
    "test_known_species_known_family": "test_known_species_known_family",
    "test_known_species_heldout_family": "test_known_species_heldout_family",
    "test_heldout_species_known_family": "test_heldout_species_known_family",
    "test_heldout_species_heldout_family": "test_heldout_species_heldout_family",
}
SAMPLE_FIELDS = ["id", "species", "sequence_rna", "structure", "mfe"]

DESCRIPTION = """# {name}

Binary precursor-miRNA classification dataset exported from the pre-miRBench
MirGeneDB 71-species benchmark.

## Task

Predict whether each RNA window is a precursor-miRNA window.

- `label = 1`: 200-nt window around the hairpin of a MirGeneDB precursor.
- `label = 0`: 200-nt genomic window around a predicted hairpin, from a
  candidate pool that excludes annotated precursors.

Both classes use the same window rule: the window is centred on the terminal
loop of the hairpin, then shifted by at most 10 nt.

## Splits

Agent-visible splits:

- `train`: pre-miRBench `train`
- `validation`: pre-miRBench `valid`

## Layout

Each split folder contains `labels.csv` and `input/samples.tsv`.

## `labels.csv`

| column | meaning |
|---|---|
| `id` | Sample ID. Joins to `samples.tsv.id`. |
| `label` | `1` precursor-miRNA window, `0` negative hairpin window. |

## `samples.tsv`

| column | meaning |
|---|---|
| `id` | Neutral split-local sample ID matching `labels.csv.id`. |
| `species` | MirGeneDB species code. |
| `sequence_rna` | Full 200-nt RNA window sequence. |
| `structure` | RNAfold dot-bracket structure for the 200-nt window. |
| `mfe` | RNAfold minimum free energy for the 200-nt window. |

Record IDs, genomic coordinates, MirGeneDB precursor IDs, family IDs,
negative-mining scores, duplicate-control keys and split reasons are not
provided; they are benchmark provenance fields and some encode label provenance.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--output-dir", required=True, help="Agentomics dataset directory, e.g. <agentomics>/datasets/<name>.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--id-map", required=True, help="Private TSV mapping exported sample IDs to dataset record IDs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the within-split row shuffle.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    grouped = defaultdict(list)
    with open(args.dataset_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[EXPORT_SPLITS[row["split"]]].append(row)

    with open(args.id_map, "w", newline="") as map_handle:
        id_map = csv.writer(map_handle, delimiter="\t")
        id_map.writerow(["export_split", "sample_id", "record_id"])
        for split, rows in grouped.items():
            random.Random(f"{args.seed}-{split}").shuffle(rows)
            input_dir = out / split / "input"
            input_dir.mkdir(parents=True)
            with open(out / split / "labels.csv", "w", newline="") as labels, open(input_dir / "samples.tsv", "w", newline="") as samples:
                label_writer = csv.writer(labels)
                sample_writer = csv.writer(samples, delimiter="\t")
                label_writer.writerow(["id", "label"])
                sample_writer.writerow(SAMPLE_FIELDS)
                for index, row in enumerate(rows, start=1):
                    sample_id = f"{split}_{index:06d}"
                    label_writer.writerow([sample_id, row["label"]])
                    sample_writer.writerow([sample_id, row["species"], row["sequence"], row["structure"], row["mfe"]])
                    id_map.writerow([split, sample_id, row["record_id"]])

    (out / "metadata.json").write_text(json.dumps({
        "name": args.dataset_name,
        "task_type": "classification",
        "positive_class": "1",
        "negative_class": "0",
        "label_to_scalar": {"0": 0, "1": 1},
        "source": "pre-miRBench MirGeneDB 71-species precursor benchmark",
    }, indent=2) + "\n")
    (out / "dataset_description.md").write_text(DESCRIPTION.format(name=args.dataset_name))
    print(json.dumps({split: {"records": len(rows), "positives": sum(r["label"] == "1" for r in rows)} for split, rows in grouped.items()}, indent=2))


if __name__ == "__main__":
    main()
