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


def phact_family_from_mature_id(mature_id):
    core = mature_id
    if core.startswith("Hsa-"):
        core = core[4:]
    core = core.rsplit("_", 1)[0]
    for marker in ("-P", "-v"):
        if marker in core:
            core = core.split(marker, 1)[0]
    return core.upper()


def premirna_id_from_mature_id(mature_id):
    return mature_id.rsplit("_", 1)[0]


def arm_from_mature_id(mature_id):
    return mature_id.rsplit("_", 1)[1] if "_" in mature_id else "NA"


def load_phact_profiles(phact_path):
    phact_rows = []
    profiles = defaultdict(dict)
    profile_counts = defaultdict(int)
    family_stats = defaultdict(lambda: {"sum": 0.0, "count": 0, "min": None, "max": None})

    with open(phact_path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        phact_columns = [
            field
            for field in fieldnames
            if field.startswith("phact_") and field.rsplit("_", 1)[-1] in {"A", "C", "G", "T"}
        ]
        for row in reader:
            phact_rows.append(row)
            mature_id = row["mirgenedb_mature_id"]
            position = int(row["mirna_position_1based"])
            profiles[mature_id][position] = row["actual_nt"]
            profile_counts[mature_id] += 1
            family = phact_family_from_mature_id(mature_id)
            for column in phact_columns:
                raw_value = row[column]
                if raw_value in {"", "NA"}:
                    continue
                value = float(raw_value)
                model, nucleotide = column[len("phact_") :].rsplit("_", 1)
                stat = family_stats[(family, model, nucleotide)]
                stat["sum"] += value
                stat["count"] += 1
                stat["min"] = value if stat["min"] is None else min(stat["min"], value)
                stat["max"] = value if stat["max"] is None else max(stat["max"], value)

    return phact_rows, profiles, profile_counts, family_stats


def write_phact_profiles(path, profiles, profile_counts):
    fields = [
        "mirgenedb_mature_id",
        "mirgenedb_premirna_id",
        "family",
        "arm",
        "mature_sequence_dna",
        "mature_length",
        "phact_position_count",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for mature_id in sorted(profiles):
            positions = profiles[mature_id]
            sequence = "".join(positions[index] for index in sorted(positions))
            writer.writerow(
                {
                    "mirgenedb_mature_id": mature_id,
                    "mirgenedb_premirna_id": premirna_id_from_mature_id(mature_id),
                    "family": phact_family_from_mature_id(mature_id),
                    "arm": arm_from_mature_id(mature_id),
                    "mature_sequence_dna": dna_sequence(sequence),
                    "mature_length": len(sequence),
                    "phact_position_count": profile_counts[mature_id],
                }
            )


def write_phact_family_summary(path, family_stats):
    fields = ["family", "phact_model", "nucleotide", "mean_score", "min_score", "max_score", "value_count"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for (family, model, nucleotide), stat in sorted(family_stats.items()):
            count = stat["count"]
            writer.writerow(
                {
                    "family": family,
                    "phact_model": model,
                    "nucleotide": nucleotide,
                    "mean_score": f"{stat['sum'] / count:.8f}" if count else "NA",
                    "min_score": f"{stat['min']:.8f}" if stat["min"] is not None else "NA",
                    "max_score": f"{stat['max']:.8f}" if stat["max"] is not None else "NA",
                    "value_count": count,
                }
            )


def write_metadata(dataset_dir, dataset_name):
    metadata = {
        "name": dataset_name,
        "task_type": "classification",
        "positive_class": "1",
        "negative_class": "0",
        "label_to_scalar": {"0": 0, "1": 1},
        "source": "pre-miRBench MirGeneDB 71-species precursor benchmark with global PHACT miRNA reference tables",
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

## Input Files

Every split has the same top-level input files:

- `samples.tsv`: one row per sample ID.
- `phact_mirna_positions.tsv`: global human mature-miRNA PHACT reference table.
- `phact_mirna_profiles.tsv`: mature-miRNA profile summary reconstructed from
  the PHACT position table.
- `phact_family_summary.tsv`: family-level summary of the PHACT score table.

## `samples.tsv`

Columns:

- `id`: neutral split-local sample ID matching `labels.csv.id`.
- `species`: MirGeneDB species code.
- `chrom`, `start_0based`, `end_0based`, `strand`: genomic window coordinates.
- `window_length`: sequence length.
- `sequence_rna`: RNA window sequence.
- `sequence_dna`: same sequence with U converted to T.
- `canonical_100nt_sequence_rna`: centered 100-nt sequence used for leakage
  control in pre-miRBench.
- `canonical_100nt_sequence_dna`: DNA alphabet version of the 100-nt sequence.
- `structure`: RNAfold dot-bracket structure for the full window.
- `mfe`: RNAfold minimum free energy.
- `gc_fraction`: GC fraction of the full window.

The exported sample table intentionally omits original pre-miRBench record IDs,
MirGeneDB precursor IDs, family IDs, negative-mining scores, and split reasons
because those fields encode label provenance and would leak the answer.

## PHACT Reference Tables

PHACT stands for PHylogeny-Aware Computation of Tolerance for nucleotide
substitutions. The included PHACT table is the same human mature-miRNA reference
used by `manakov_phact/train/input/phact_mirna_positions.tsv`.

`phact_mirna_positions.tsv` is keyed by `mirgenedb_mature_id` and
`mirna_position_1based`; it is not keyed by sample ID. It should be treated as a
global sequence/evolutionary reference. Each position contains the observed
nucleotide plus nucleotide-state scores for multiple PHACT models.

`phact_mirna_profiles.tsv` provides one row per mature miRNA with the mature
sequence reconstructed from the PHACT position rows.

`phact_family_summary.tsv` aggregates PHACT scores by mature-miRNA family,
PHACT model, and nucleotide.
"""
    (dataset_dir / "dataset_description.md").write_text(description)


def reset_dir(path, overwrite):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing directory without --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def make_split(split_dir, rows, phact_path, phact_profiles_path, phact_family_summary_path):
    input_dir = split_dir / "input"
    input_dir.mkdir(parents=True)
    shutil.copy2(phact_path, input_dir / "phact_mirna_positions.tsv")
    shutil.copy2(phact_profiles_path, input_dir / "phact_mirna_profiles.tsv")
    shutil.copy2(phact_family_summary_path, input_dir / "phact_family_summary.tsv")

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
- PHACT miRNA positions source: `{source_phact_path}`
"""
    (supp_dir / "README.md").write_text(readme)


def main():
    parser = argparse.ArgumentParser(description="Export pre-miRBench as an Agentomics dataset.")
    parser.add_argument("--dataset-csv", default="/SCRATCH/dtzim01/pre-miRBench/datasets/mirgenedb_71/dataset.csv")
    parser.add_argument(
        "--phact-mirna-positions",
        default="/home/dtzim01/agentomics-ml/datasets/manakov_phact/train/input/phact_mirna_positions.tsv",
    )
    parser.add_argument("--agentomics-root", default="/home/dtzim01/agentomics-ml")
    parser.add_argument("--dataset-name", default="premirbench_mirgenedb71")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_csv = Path(args.dataset_csv)
    phact_path = Path(args.phact_mirna_positions)
    agentomics_root = Path(args.agentomics_root)
    public_dir = agentomics_root / "datasets" / args.dataset_name
    hidden_dir = agentomics_root / "test_datasets" / args.dataset_name

    if not dataset_csv.is_file():
        raise FileNotFoundError(dataset_csv)
    if not phact_path.is_file():
        raise FileNotFoundError(phact_path)

    reset_dir(public_dir, args.overwrite)
    reset_dir(hidden_dir, args.overwrite)

    _, profiles, profile_counts, family_stats = load_phact_profiles(phact_path)
    phact_work_dir = public_dir / "supplementary" / "_phact_export"
    phact_work_dir.mkdir(parents=True)
    phact_profiles_path = phact_work_dir / "phact_mirna_profiles.tsv"
    phact_family_summary_path = phact_work_dir / "phact_family_summary.tsv"
    write_phact_profiles(phact_profiles_path, profiles, profile_counts)
    write_phact_family_summary(phact_family_summary_path, family_stats)

    grouped = grouped_rows(dataset_csv)
    for split_name in ("train", "validation"):
        make_split(public_dir / split_name, grouped[split_name], phact_path, phact_profiles_path, phact_family_summary_path)
    for split_name in sorted(TEST_SPLITS.values()):
        make_split(hidden_dir / split_name, grouped[split_name], phact_path, phact_profiles_path, phact_family_summary_path)

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
    print(json.dumps({"dataset": args.dataset_name, "splits": summary, "phact_mature_profiles": len(profiles)}, indent=2))


if __name__ == "__main__":
    main()
