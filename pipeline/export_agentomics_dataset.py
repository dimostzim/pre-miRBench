#!/usr/bin/env python
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


PUBLIC_SPLITS = {
    "train": "train",
    "valid": "validation",
}

TEST_SPLITS = {
    "test_known_species_known_family": "test_known_species_known_family",
    "test_known_species_heldout_family": "test_known_species_heldout_family",
    "test_heldout_species_known_family": "test_heldout_species_known_family",
    "test_heldout_species_heldout_family": "test_heldout_species_heldout_family",
}

DEFAULT_PHACT_PREMIRNA_POSITIONS = (
    "/home/dtzim01/drive-download-19Ntprvu-qbI1k4ZQphZ4QnuFoXNIgK2E/"
    "extracted/results_0226/orthologs_qntnorm_transformed.tsv"
)

SAMPLE_FIELDS = [
    "id",
    "species",
    "chrom",
    "start_0based",
    "end_0based",
    "strand",
    "window_length",
    "sequence_rna",
    "sequence_dna",
    "canonical_100nt_sequence_rna",
    "canonical_100nt_sequence_dna",
    "structure",
    "mfe",
    "gc_fraction",
]


def dna_sequence(sequence):
    return sequence.upper().replace("U", "T")


def gc_fraction(sequence):
    seq = dna_sequence(sequence)
    if not seq:
        return ""
    gc = seq.count("G") + seq.count("C")
    return f"{gc / len(seq):.6f}"


def phact_family_from_premirna_id(premirna_id):
    core = premirna_id
    if core.startswith("Hsa-"):
        core = core[4:]
    for marker in ("-P", "-v"):
        if marker in core:
            core = core.split(marker, 1)[0]
    return core.upper()


def load_premirna_phact_index(phact_path):
    position_nucleotides = defaultdict(lambda: defaultdict(set))
    row_count = 0

    with open(phact_path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required_fields = {"ID", "Position", "Nucleotide"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"PHACT precursor table missing required columns: {missing}")

        for row in reader:
            row_count += 1
            premirna_id = row["ID"]
            position = int(row["Position"])
            nucleotide = row["Nucleotide"].upper()
            position_nucleotides[premirna_id][position].add(nucleotide)

    index_rows = []
    for premirna_id in sorted(position_nucleotides):
        positions = sorted(position_nucleotides[premirna_id])
        all_four_states = all(
            position_nucleotides[premirna_id][position] == {"A", "C", "G", "T"} for position in positions
        )
        index_rows.append(
            {
                "mirgenedb_premirna_id": premirna_id,
                "family": phact_family_from_premirna_id(premirna_id),
                "position_count": len(positions),
                "min_position": min(positions),
                "max_position": max(positions),
                "has_all_four_nt_states": "true" if all_four_states else "false",
            }
        )

    return index_rows, row_count


def write_premirna_phact_index(path, index_rows):
    fields = [
        "mirgenedb_premirna_id",
        "family",
        "position_count",
        "min_position",
        "max_position",
        "has_all_four_nt_states",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(index_rows)


def write_metadata(dataset_dir, dataset_name):
    metadata = {
        "name": dataset_name,
        "task_type": "classification",
        "positive_class": "1",
        "negative_class": "0",
        "label_to_scalar": {"0": 0, "1": 1},
        "source": (
            "pre-miRBench MirGeneDB 71-species precursor benchmark with "
            "global precursor-level PHACT reference table"
        ),
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def write_description(dataset_dir, dataset_name):
    description = f"""# {dataset_name}

Binary precursor-miRNA classification dataset exported from the pre-miRBench
MirGeneDB 71-species benchmark.

## Task

Predict whether each RNA window is a precursor-miRNA candidate.

- `label = 1`: positive precursor-miRNA window from MirGeneDB.
- `label = 0`: hard negative hairpin-like genomic window.

The prediction unit is `id`. Labels are stored in each split's `labels.csv`.
Inputs are stored under each split's `input/` directory.

This is a sequence classification task. A model should use the provided RNA
window sequence and optional derived features/reference tables to predict the
binary label for each sample ID.

## Splits

Public Agentomics splits:

- `train`: pre-miRBench `train`
- `validation`: pre-miRBench `valid`

Hidden Agentomics test splits:

- `test_known_species_known_family`
- `test_known_species_heldout_family`
- `test_heldout_species_known_family`
- `test_heldout_species_heldout_family`

These hidden splits are stored under `test_datasets/{dataset_name}/`.

## Layout

Each split folder contains:

```text
labels.csv
input/
  samples.tsv
  phact_premirna_positions.tsv
  phact_premirna_index.tsv
```

Every split has the same top-level input files under `input/`:

- `samples.tsv`: one row per sample ID.
- `phact_premirna_positions.tsv`: global human precursor-miRNA PHACT reference
  table.
- `phact_premirna_index.tsv`: small index of precursor IDs, families, and
  precursor-position counts from the PHACT table.

Each split also has `labels.csv` outside `input/`.

## `labels.csv`

| column | meaning |
|---|---|
| `id` | Sample ID. Joins to `samples.tsv.id`. |
| `label` | Binary class label. `1` means precursor-miRNA, `0` means negative hairpin-like genomic window. |

## `samples.tsv`

One row per precursor-miRNA candidate window.

| column | meaning |
|---|---|
| `id` | Neutral split-local sample ID matching `labels.csv.id`. |
| `species` | MirGeneDB species code. |
| `chrom` | Sequence/chromosome name in the combined benchmark genome. |
| `start_0based` | 0-based inclusive window start coordinate. |
| `end_0based` | 0-based exclusive window end coordinate. |
| `strand` | Genomic strand. |
| `window_length` | Full RNA window length. |
| `sequence_rna` | Full RNA window sequence. |
| `sequence_dna` | Same sequence with U converted to T. |
| `canonical_100nt_sequence_rna` | Centered 100-nt sequence used for leakage control in pre-miRBench. |
| `canonical_100nt_sequence_dna` | DNA alphabet version of the centered 100-nt sequence. |
| `structure` | RNAfold dot-bracket structure for the full window. |
| `mfe` | RNAfold minimum free energy for the full window. |
| `gc_fraction` | GC fraction of the full window. |

The exported sample table intentionally omits original pre-miRBench record IDs,
MirGeneDB precursor IDs, family IDs, negative-mining scores, and split reasons
because those fields encode label provenance and would leak the answer.

## PHACT Reference Tables

PHACT stands for PHylogeny-Aware Computation of Tolerance for nucleotide
substitutions. The included table is the human precursor-miRNA PHACT reference
from `orthologs_qntnorm_transformed.tsv`.

`phact_premirna_positions.tsv` is keyed by MirGeneDB precursor ID (`ID`),
precursor position (`Position`), and nucleotide state (`Nucleotide`); it is not
keyed by sample ID. It should be treated as a global sequence/evolutionary
reference. Each precursor position has A/C/G/T score rows for multiple PHACT
models.

| column | meaning |
|---|---|
| `ID` | Human MirGeneDB precursor-miRNA ID, for example `Hsa-Let-7-P1b`. |
| `Position` | 1-based position within the precursor-miRNA sequence. |
| `Nucleotide` | Candidate nucleotide state at that precursor position. |
| `PHACTn_*` | PHACT tolerance score from one normalization/model variant. |

`phact_premirna_index.tsv` provides one row per precursor ID so agents can see
which precursor families and lengths are represented without scanning the full
PHACT score table.

| column | meaning |
|---|---|
| `mirgenedb_premirna_id` | Human MirGeneDB precursor-miRNA ID. |
| `family` | Precursor family parsed from the MirGeneDB precursor ID. |
| `position_count` | Number of precursor positions in the PHACT table. |
| `min_position` | Minimum observed 1-based position. |
| `max_position` | Maximum observed 1-based position. |
| `has_all_four_nt_states` | Whether every precursor position has A/C/G/T rows. |

## Joins

```text
labels.csv.id -> samples.tsv.id
```

The PHACT reference tables do not join to `samples.tsv` by sample ID. They are
global human precursor-miRNA reference tables.

## PHACT Scores

PHACTn stands for PHylogeny-Aware Computation of Tolerance for Nucleotide
substitutions. The PHACT table provides nucleotide-specific evolutionary
features derived from orthologous precursor-miRNA alignments and phylogenetic
context.

Conceptually, PHACT asks how tolerant an aligned precursor position appears to
be to different nucleotide states, given the pattern seen across orthologues
and the phylogenetic relationships among species. Positions that are strongly
preserved across evolution, or where alternative nucleotide states are less
compatible with the phylogenetic pattern, can be interpreted as more
evolutionarily constrained.

Each precursor position has four rows, one for each possible nucleotide state:
`A`, `C`, `G`, and `T`. The dataset keeps all model-specific PHACT columns from
the source table rather than reducing them to one summary score.

## Missing Values

- `samples.tsv`: no MirGeneDB precursor IDs, family IDs, negative-mining
  scores, or split-reason fields are provided, because those fields would leak
  label provenance.
- `phact_premirna_positions.tsv`: only human precursor-miRNA PHACT rows from
  the source table are included.
- `phact_premirna_index.tsv`: `has_all_four_nt_states=true` means every
  precursor position has A/C/G/T rows in the score table.

The PHACT files are optional global reference data. They are not a lookup table
for every benchmark sample: the benchmark contains 71 species, while this PHACT
reference is human precursor-level data. The sample table deliberately does not
include MirGeneDB precursor IDs or family IDs, because those fields identify
positive examples and would leak the label.
"""
    (dataset_dir / "dataset_description.md").write_text(description)


def reset_dir(path, overwrite):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing directory without --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def make_split(split_dir, rows, phact_path, phact_index_path):
    input_dir = split_dir / "input"
    input_dir.mkdir(parents=True)
    shutil.copy2(phact_path, input_dir / "phact_premirna_positions.tsv")
    shutil.copy2(phact_index_path, input_dir / "phact_premirna_index.tsv")

    labels_path = split_dir / "labels.csv"
    samples_path = input_dir / "samples.tsv"
    with open(labels_path, "w", newline="") as labels_handle, open(samples_path, "w", newline="") as samples_handle:
        label_writer = csv.DictWriter(labels_handle, fieldnames=["id", "label"])
        sample_writer = csv.DictWriter(samples_handle, fieldnames=SAMPLE_FIELDS, delimiter="\t")
        label_writer.writeheader()
        sample_writer.writeheader()
        for row in rows:
            sample_id = row["agentomics_id"]
            sequence = row["sequence"]
            canonical = row["canonical_100nt_sequence"]
            label_writer.writerow({"id": sample_id, "label": row["label"]})
            sample_writer.writerow(
                {
                    "id": sample_id,
                    "species": row["species"],
                    "chrom": row["chrom"],
                    "start_0based": row["start"],
                    "end_0based": row["end"],
                    "strand": row["strand"],
                    "window_length": len(sequence),
                    "sequence_rna": sequence,
                    "sequence_dna": dna_sequence(sequence),
                    "canonical_100nt_sequence_rna": canonical,
                    "canonical_100nt_sequence_dna": dna_sequence(canonical),
                    "structure": row["structure"],
                    "mfe": row["mfe"],
                    "gc_fraction": gc_fraction(sequence),
                }
            )


def grouped_rows(dataset_csv):
    grouped = defaultdict(list)
    counters = defaultdict(int)
    with open(dataset_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_split = row["split"]
            if source_split in PUBLIC_SPLITS:
                export_split = PUBLIC_SPLITS[source_split]
            elif source_split in TEST_SPLITS:
                export_split = TEST_SPLITS[source_split]
            else:
                raise ValueError(f"Unexpected split: {source_split}")
            counters[export_split] += 1
            row = dict(row)
            row["agentomics_id"] = f"{export_split}_{counters[export_split]:06d}"
            grouped[export_split].append(row)
    return grouped


def write_supplementary(dataset_dir, source_dataset_csv, source_phact_path):
    supp_dir = dataset_dir / "supplementary"
    supp_dir.mkdir(exist_ok=True)
    readme = f"""# Supplementary

This folder records source-level provenance only. It intentionally does not
contain per-sample source IDs because the original benchmark IDs encode
positive/negative provenance.

- pre-miRBench dataset CSV: `{source_dataset_csv}`
- PHACT precursor positions source: `{source_phact_path}`
"""
    (supp_dir / "README.md").write_text(readme)


def main():
    parser = argparse.ArgumentParser(description="Export pre-miRBench as an Agentomics dataset.")
    parser.add_argument("--dataset-csv", default="/SCRATCH/dtzim01/pre-miRBench/datasets/mirgenedb_71/dataset.csv")
    parser.add_argument(
        "--phact-premirna-positions",
        default=DEFAULT_PHACT_PREMIRNA_POSITIONS,
    )
    parser.add_argument("--agentomics-root", default="/home/dtzim01/agentomics-ml")
    parser.add_argument("--dataset-name", default="premirbench_mirgenedb71")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_csv = Path(args.dataset_csv)
    phact_path = Path(args.phact_premirna_positions)
    agentomics_root = Path(args.agentomics_root)
    public_dir = agentomics_root / "datasets" / args.dataset_name
    hidden_dir = agentomics_root / "test_datasets" / args.dataset_name

    if not dataset_csv.is_file():
        raise FileNotFoundError(dataset_csv)
    if not phact_path.is_file():
        raise FileNotFoundError(phact_path)

    reset_dir(public_dir, args.overwrite)
    reset_dir(hidden_dir, args.overwrite)

    phact_index_rows, phact_row_count = load_premirna_phact_index(phact_path)
    phact_work_dir = public_dir / "supplementary" / "_phact_export"
    phact_work_dir.mkdir(parents=True)
    phact_index_path = phact_work_dir / "phact_premirna_index.tsv"
    write_premirna_phact_index(phact_index_path, phact_index_rows)

    grouped = grouped_rows(dataset_csv)
    for split_name in ("train", "validation"):
        make_split(public_dir / split_name, grouped[split_name], phact_path, phact_index_path)
    for split_name in sorted(TEST_SPLITS.values()):
        make_split(hidden_dir / split_name, grouped[split_name], phact_path, phact_index_path)

    shutil.rmtree(phact_work_dir)
    write_metadata(public_dir, args.dataset_name)
    write_description(public_dir, args.dataset_name)
    write_supplementary(public_dir, dataset_csv, phact_path)

    summary = {
        split: {
            "records": len(rows),
            "positives": sum(1 for row in rows if row["label"] == "1"),
            "negatives": sum(1 for row in rows if row["label"] == "0"),
        }
        for split, rows in sorted(grouped.items())
    }
    print(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "splits": summary,
                "phact_premirnas": len(phact_index_rows),
                "phact_rows": phact_row_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
