#!/usr/bin/env python3
"""Lazy, deterministic structure-aware representation for pre-miRBench."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEQ_CHANNELS = {"A": 0, "C": 1, "G": 2, "U": 3}
STRUCT_CHANNELS = {".": 4, "(": 5, ")": 6}
PARTNER_SEQUENCE_CHANNELS = {"A": 7, "C": 8, "G": 9, "U": 10}
PARTNER_OFFSET_CHANNEL = 11
EXPECTED_COLUMNS = ["id", "species", "sequence_rna", "structure", "mfe"]


@dataclass(frozen=True)
class RepresentationMetadata:
    version: int
    length: int
    num_channels: int
    sequence_channels: dict[str, int]
    structure_channels: dict[str, int]
    partner_sequence_channels: dict[str, int]
    partner_offset_channel: int
    partner_offset_scale: float
    unpaired_partner_index: str
    unknown_sequence_symbol: str
    unknown_encoding: str
    sequence_normalization: str
    mfe_mean: float
    mfe_std: float
    mfe_std_definition: str
    label_column: str
    uses_species: bool
    uses_id_as_feature: bool

    @classmethod
    def fit(cls, train_input: Union[str, Path]) -> "RepresentationMetadata":
        frame = read_samples(train_input)
        mfe = frame["mfe"].to_numpy(dtype=np.float64)
        mean, std = float(mfe.mean()), float(mfe.std(ddof=0))
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid training MFE moments: mean={mean}, std={std}")
        return cls(
            version=2, length=200, num_channels=12,
            sequence_channels=dict(SEQ_CHANNELS),
            structure_channels=dict(STRUCT_CHANNELS),
            partner_sequence_channels=dict(PARTNER_SEQUENCE_CHANNELS),
            partner_offset_channel=PARTNER_OFFSET_CHANNEL,
            partner_offset_scale=199.0,
            unpaired_partner_index="self",
            unknown_sequence_symbol="N",
            unknown_encoding="zeros in own and partner nucleotide one-hot channels",
            sequence_normalization="strip whitespace, uppercase, T->U",
            mfe_mean=mean, mfe_std=std,
            mfe_std_definition="population standard deviation (ddof=0), training split only",
            label_column="numeric_label", uses_species=False, uses_id_as_feature=False,
        )

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RepresentationMetadata":
        return cls(**json.loads(Path(path).read_text()))


def _samples_path(input_path: Union[str, Path]) -> Path:
    path = Path(input_path)
    if path.is_dir():
        path = path / "samples.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"samples.tsv not found at {path}")
    return path


def normalize_sequence(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("sequence_rna must be a string")
    return "".join(value.split()).upper().replace("T", "U")


def reverse_complement_raw(sequence: str, structure: str) -> tuple[str, str]:
    """Deterministic raw-level reverse complement preserving dot-bracket pairing."""
    sequence = normalize_sequence(sequence)
    validate_sequence(sequence, len(sequence))
    structure = validate_structure(structure, len(sequence))
    complement = str.maketrans({"A": "U", "U": "A", "C": "G", "G": "C", "N": "N"})
    rc_sequence = sequence[::-1].translate(complement)
    rc_structure = structure[::-1].translate(str.maketrans({"(": ")", ")": "("}))
    validate_structure(rc_structure, len(sequence))
    return rc_sequence, rc_structure


def validate_sequence(sequence: str, expected_length: int = 200) -> None:
    if len(sequence) != expected_length:
        raise ValueError(f"RNA sequence length {len(sequence)} != {expected_length}")
    invalid = set(sequence) - set("ACGUN")
    if invalid:
        raise ValueError(f"Invalid RNA symbols: {sorted(invalid)}")


def parse_partner_indices(structure: object, expected_length: int = 200) -> np.ndarray:
    """Validate dot-bracket notation and return exact partner index, or self if unpaired."""
    if not isinstance(structure, str):
        raise ValueError("structure must be a string")
    structure = structure.strip()
    if len(structure) != expected_length:
        raise ValueError(f"Structure length {len(structure)} != {expected_length}")
    invalid = set(structure) - set(".()")
    if invalid:
        raise ValueError(f"Invalid structure symbols: {sorted(invalid)}")
    partners = np.arange(expected_length, dtype=np.int64)
    stack: list[int] = []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ValueError(f"Unbalanced structure: unmatched ')' at index {index}")
            left = stack.pop()
            partners[left], partners[index] = index, left
    if stack:
        raise ValueError(f"Unbalanced structure: unmatched '(' at index {stack[-1]}")
    return partners


def validate_structure(structure: object, expected_length: int = 200) -> str:
    parse_partner_indices(structure, expected_length)
    return structure.strip()  # type: ignore[union-attr]


def read_samples(input_path: Union[str, Path]) -> pd.DataFrame:
    path = _samples_path(input_path)
    frame = pd.read_csv(path, sep="\t", dtype={"id": "string", "species": "string"})
    missing = [c for c in EXPECTED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required sample columns: {missing}")
    frame = frame[EXPECTED_COLUMNS].copy()
    if frame["id"].isna().any() or (frame["id"].str.len() == 0).any():
        raise ValueError("Sample IDs must be nonmissing and nonempty")
    if frame["id"].duplicated().any():
        raise ValueError("Sample IDs must be unique")
    frame["sequence_rna"] = frame["sequence_rna"].map(normalize_sequence)
    for sequence in frame["sequence_rna"]:
        validate_sequence(sequence)
    frame["structure"] = frame["structure"].map(lambda x: validate_structure(x))
    frame["mfe"] = pd.to_numeric(frame["mfe"], errors="coerce")
    if not np.isfinite(frame["mfe"].to_numpy(dtype=np.float64)).all():
        raise ValueError("MFE values must be finite numeric values")
    return frame


def attach_labels(samples: pd.DataFrame, labels_path: Union[str, Path], label_column: str) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, dtype={"id": "string"})
    required = {"id", label_column}
    if not required.issubset(labels.columns):
        raise ValueError(f"labels.csv must contain {sorted(required)}")
    labels = labels[["id", label_column]].copy()
    if labels["id"].isna().any() or labels["id"].duplicated().any():
        raise ValueError("Label IDs must be nonmissing and unique")
    labels[label_column] = pd.to_numeric(labels[label_column], errors="coerce")
    if not labels[label_column].isin([0, 1]).all():
        raise ValueError("Labels must be binary 0/1")
    if set(samples["id"]) != set(labels["id"]):
        raise ValueError("Sample and label ID sets differ")
    merged = samples.merge(labels, on="id", how="left", validate="one_to_one", sort=False)
    if len(merged) != len(samples):
        raise RuntimeError("One-to-one label join changed row count")
    return merged


def encode_representation(sequence: str, structure: str, length: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return 12xL float channels and L int64 partner indices."""
    sequence = normalize_sequence(sequence)
    validate_sequence(sequence, length)
    structure = validate_structure(structure, length)
    partners = parse_partner_indices(structure, length)
    encoded = np.zeros((12, length), dtype=np.float32)
    positions = np.arange(length)

    own = np.fromiter((SEQ_CHANNELS.get(x, -1) for x in sequence), dtype=np.int8, count=length)
    known = own >= 0
    encoded[own[known], positions[known]] = 1.0
    struct = np.fromiter((STRUCT_CHANNELS[x] for x in structure), dtype=np.int8, count=length)
    encoded[struct, positions] = 1.0

    paired = partners != positions
    paired_positions = positions[paired]
    partner_symbols = [sequence[partners[i]] for i in paired_positions]
    partner_channels = np.fromiter(
        (PARTNER_SEQUENCE_CHANNELS.get(x, -1) for x in partner_symbols),
        dtype=np.int8, count=len(paired_positions),
    )
    partner_known = partner_channels >= 0
    encoded[partner_channels[partner_known], paired_positions[partner_known]] = 1.0
    encoded[PARTNER_OFFSET_CHANNEL, paired] = (
        (partners[paired] - positions[paired]).astype(np.float32) / float(length - 1)
    )
    return encoded, partners


def encode_channels(sequence: str, structure: str, length: int = 200) -> np.ndarray:
    """Compatibility helper returning only the position-wise channels."""
    return encode_representation(sequence, structure, length)[0]


class PreMiRNADataset(Dataset):
    """Lazy fixed-shape dataset; IDs/species remain metadata, never model features."""
    def __init__(self, input_path: Union[str, Path], metadata: RepresentationMetadata,
                 labels_path: Optional[Union[str, Path]] = None) -> None:
        self.metadata = metadata
        self.frame = read_samples(input_path)
        if labels_path is not None:
            self.frame = attach_labels(self.frame, labels_path, metadata.label_column)
        self.has_labels = labels_path is not None

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def ids(self) -> list[str]:
        return self.frame["id"].astype(str).tolist()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        channels, partners = encode_representation(row.sequence_rna, row.structure, self.metadata.length)
        mfe = (float(row.mfe) - self.metadata.mfe_mean) / self.metadata.mfe_std
        item = {
            "channels": torch.from_numpy(channels),
            "partner_indices": torch.from_numpy(partners),
            "mfe": torch.tensor([mfe], dtype=torch.float32),
        }
        if self.has_labels:
            item["label"] = torch.tensor(float(row[self.metadata.label_column]), dtype=torch.float32)
        return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit MFE scaling and check representation")
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--check-input", required=True)
    parser.add_argument("--check-labels")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metadata = RepresentationMetadata.fit(args.train_input)
    metadata.save(args.output)
    dataset = PreMiRNADataset(args.check_input, metadata, args.check_labels)
    first = dataset[0]
    result = {
        "rows": len(dataset), "channels_shape": list(first["channels"].shape),
        "partner_indices_shape": list(first["partner_indices"].shape),
        "channels_dtype": str(first["channels"].dtype),
        "partner_indices_dtype": str(first["partner_indices"].dtype),
        "mfe_shape": list(first["mfe"].shape), "mfe_dtype": str(first["mfe"].dtype),
        "mfe_mean": metadata.mfe_mean, "mfe_std": metadata.mfe_std,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
