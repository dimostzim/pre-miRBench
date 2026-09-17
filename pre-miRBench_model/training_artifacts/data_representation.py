#!/usr/bin/env python3
"""Lazy species-conditioned, pair-aware representation for pre-miRBench."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LENGTH = 200
SEQ_CHANNELS = {"A": 0, "C": 1, "G": 2, "U": 3}
STRUCT_CHANNELS = {".": 4, "(": 5, ")": 6}
PARTNER_SEQUENCE_CHANNELS = {"A": 7, "C": 8, "G": 9, "U": 10}
PARTNER_OFFSET_CHANNEL = 11
BASES = frozenset("ACGUN")
PAIR_TYPES = ("AU", "UA", "GC", "CG", "GU", "UG")
GLOBAL_FEATURES = (
    "mfe", "gc_fraction", "paired_fraction", "mean_pair_span",
    "maximum_pair_span", "maximum_nesting_depth",
    "maximum_contiguous_unpaired_run", "pair_fraction_AU", "pair_fraction_UA",
    "pair_fraction_GC", "pair_fraction_CG", "pair_fraction_GU", "pair_fraction_UG",
)
EXPECTED_COLUMNS = ("id", "species", "sequence_rna", "structure", "mfe")
RNA_COMPLEMENT = str.maketrans({"A": "U", "U": "A", "C": "G", "G": "C", "N": "N"})
BRACKET_SWAP = str.maketrans({"(": ")", ")": "(", ".": "."})


def normalize_sequence(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("sequence_rna must be a string")
    sequence = "".join(value.split()).upper().replace("T", "U")
    invalid = sorted(set(sequence) - BASES)
    if invalid:
        raise ValueError(f"Invalid RNA symbols: {invalid}")
    return sequence


def validate_sequence(sequence: str, expected_length: int = LENGTH) -> None:
    if len(sequence) != expected_length:
        raise ValueError(f"Sequence length {len(sequence)} != {expected_length}")
    invalid = sorted(set(sequence) - BASES)
    if invalid:
        raise ValueError(f"Invalid RNA symbols: {invalid}")


def parse_structure(structure: Any, expected_length: int = LENGTH) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """Return sorted (left, right, opening_depth) pairs and self-indexed partners."""
    if not isinstance(structure, str):
        raise ValueError("structure must be a string")
    structure = "".join(structure.split())
    if len(structure) != expected_length:
        raise ValueError(f"Structure length {len(structure)} != {expected_length}")
    invalid = sorted(set(structure) - set(".()"))
    if invalid:
        raise ValueError(f"Invalid structure symbols: {invalid}")
    stack: list[tuple[int, int]] = []
    pairs: list[tuple[int, int, int]] = []
    partners = np.arange(expected_length, dtype=np.int64)
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append((index, len(stack) + 1))
        elif symbol == ")":
            if not stack:
                raise ValueError(f"Unbalanced structure: unmatched ')' at index {index}")
            left, depth = stack.pop()
            partners[left], partners[index] = index, left
            pairs.append((left, index, depth))
    if stack:
        raise ValueError(f"Unbalanced structure: unmatched '(' at index {stack[-1][0]}")
    pairs.sort(key=lambda item: item[0])
    return pairs, partners


def parse_partner_indices(structure: Any, expected_length: int = LENGTH) -> np.ndarray:
    return parse_structure(structure, expected_length)[1]


def validate_structure(structure: Any, expected_length: int = LENGTH) -> str:
    normalized = "".join(structure.split()) if isinstance(structure, str) else structure
    parse_structure(normalized, expected_length)
    return normalized


def reverse_complement_sequence(sequence: Any) -> str:
    normalized = normalize_sequence(sequence)
    validate_sequence(normalized, len(normalized))
    return normalized[::-1].translate(RNA_COMPLEMENT)


def reverse_structure(structure: Any) -> str:
    if not isinstance(structure, str):
        raise ValueError("structure must be a string")
    normalized = "".join(structure.split())
    parse_structure(normalized, len(normalized))
    return normalized[::-1].translate(BRACKET_SWAP)


def reverse_complement_view(sequence: Any, structure: Any, mfe: Any) -> tuple[str, str, float]:
    value = float(mfe)
    if not math.isfinite(value):
        raise ValueError("MFE must be finite")
    return reverse_complement_sequence(sequence), reverse_structure(structure), value


def encode_representation(sequence: str, structure: str, length: int = LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """Iteration-6-compatible 12xL channels and exact L partner indices."""
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


def encode_channels(sequence: str, structure: str, length: int = LENGTH) -> np.ndarray:
    return encode_representation(sequence, structure, length)[0]


def raw_global_features(sequence: str, structure: str, mfe: float,
                        pairs: Optional[list[tuple[int, int, int]]] = None) -> np.ndarray:
    """Compute the 13 globals with definitions exactly matching iteration 30."""
    sequence = normalize_sequence(sequence)
    validate_sequence(sequence, LENGTH)
    structure = validate_structure(structure, LENGTH)
    if pairs is None:
        pairs, _ = parse_structure(structure)
    pair_count = len(pairs)
    spans = np.asarray([right - left for left, right, _ in pairs], dtype=np.float64)
    counts = {pair_type: 0 for pair_type in PAIR_TYPES}
    for left, right, _ in pairs:
        pair_type = sequence[left] + sequence[right]
        if pair_type in counts:
            counts[pair_type] += 1
    runs = [len(run) for run in structure.split("(") for run in run.split(")")]
    denominator = float(pair_count) if pair_count else 1.0
    values = [
        float(mfe),
        (sequence.count("G") + sequence.count("C")) / LENGTH,
        (2.0 * pair_count) / LENGTH,
        float(spans.mean()) if pair_count else 0.0,
        float(spans.max()) if pair_count else 0.0,
        float(max((depth for _, _, depth in pairs), default=0)),
        float(max(runs, default=0)),
    ] + [counts[pair_type] / denominator for pair_type in PAIR_TYPES]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (13,) or not np.isfinite(result).all():
        raise ValueError("Non-finite or malformed global features")
    return result


def _samples_path(input_path: Union[str, Path]) -> Path:
    path = Path(input_path)
    if path.is_dir():
        path = path / "samples.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"samples.tsv not found at {path}")
    return path


def read_samples(input_path: Union[str, Path]) -> pd.DataFrame:
    frame = pd.read_csv(_samples_path(input_path), sep="\t", dtype={"id": str, "species": str})
    if list(frame.columns) != list(EXPECTED_COLUMNS):
        raise ValueError(f"samples.tsv columns must be exactly {list(EXPECTED_COLUMNS)}")
    if frame.empty or frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("Sample IDs must be nonmissing, unique, and nonempty")
    if (frame["id"].str.len() == 0).any() or frame["species"].isna().any() or (frame["species"].str.len() == 0).any():
        raise ValueError("IDs and species must be nonempty")
    frame = frame.copy()
    frame["mfe"] = pd.to_numeric(frame["mfe"], errors="coerce")
    if not np.isfinite(frame["mfe"].to_numpy(dtype=np.float64)).all():
        raise ValueError("MFE values must be finite numeric values")
    sequences, structures = [], []
    for sample_id, sequence, structure in frame[["id", "sequence_rna", "structure"]].itertuples(index=False, name=None):
        try:
            sequence = normalize_sequence(sequence)
            validate_sequence(sequence)
            structure = validate_structure(structure)
        except Exception as exc:
            raise ValueError(f"Invalid sample {sample_id!r}: {exc}") from exc
        sequences.append(sequence); structures.append(structure)
    frame["sequence_rna"] = sequences
    frame["structure"] = structures
    return frame


def read_labels(labels_path: Union[str, Path], sample_ids: list[str], label_column: str) -> np.ndarray:
    labels = pd.read_csv(labels_path, dtype={"id": str})
    if list(labels.columns) != ["id", label_column]:
        raise ValueError(f"labels.csv columns must be exactly ['id', '{label_column}']")
    if labels["id"].isna().any() or labels["id"].duplicated().any():
        raise ValueError("Label IDs must be nonmissing and unique")
    values = pd.to_numeric(labels[label_column], errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(f"{label_column} must contain only 0 and 1")
    mapping = dict(zip(labels["id"], values.astype(np.float32)))
    if set(mapping) != set(sample_ids):
        raise ValueError("Sample and label ID sets differ")
    return np.asarray([mapping[sample_id] for sample_id in sample_ids], dtype=np.float32)


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
    global_feature_names: list[str]
    global_feature_means: list[float]
    global_feature_stds: list[float]
    standard_deviation_definition: str
    species_vocabulary: list[str]
    unknown_species_index: int
    species_index_offset: int
    sequence_normalization: str
    unknown_sequence_symbol: str
    unknown_encoding: str
    label_column: str
    uses_species: bool
    uses_id_as_feature: bool

    @classmethod
    def fit(cls, train_input: Union[str, Path]) -> "RepresentationMetadata":
        frame = read_samples(train_input)
        values = np.stack([
            raw_global_features(sequence, structure, mfe)
            for sequence, structure, mfe in frame[["sequence_rna", "structure", "mfe"]].itertuples(index=False, name=None)
        ]).astype(np.float64)
        means = values.mean(axis=0)
        stds = values.std(axis=0, ddof=0)
        if not np.isfinite(means).all() or not np.isfinite(stds).all():
            raise ValueError("Global training moments are non-finite")
        stds[stds == 0.0] = 1.0
        vocabulary = sorted(frame["species"].unique().tolist())
        result = cls(
            version=1, length=LENGTH, num_channels=12,
            sequence_channels=dict(SEQ_CHANNELS), structure_channels=dict(STRUCT_CHANNELS),
            partner_sequence_channels=dict(PARTNER_SEQUENCE_CHANNELS),
            partner_offset_channel=PARTNER_OFFSET_CHANNEL, partner_offset_scale=199.0,
            unpaired_partner_index="self",
            global_feature_names=list(GLOBAL_FEATURES),
            global_feature_means=means.tolist(), global_feature_stds=stds.tolist(),
            standard_deviation_definition="population standard deviation (ddof=0), complete training split only; zero replaced by one",
            species_vocabulary=vocabulary, unknown_species_index=0, species_index_offset=1,
            sequence_normalization="remove whitespace, uppercase, T->U; alphabet A/C/G/U/N",
            unknown_sequence_symbol="N",
            unknown_encoding="zeros in own and partner nucleotide one-hot channels",
            label_column="numeric_label", uses_species=True, uses_id_as_feature=False,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.length != LENGTH or self.num_channels != 12:
            raise ValueError("Incompatible position representation metadata")
        if (self.sequence_channels != SEQ_CHANNELS or self.structure_channels != STRUCT_CHANNELS or
                self.partner_sequence_channels != PARTNER_SEQUENCE_CHANNELS or
                self.partner_offset_channel != PARTNER_OFFSET_CHANNEL or self.partner_offset_scale != 199.0):
            raise ValueError("Position channel definitions differ from implementation")
        if self.global_feature_names != list(GLOBAL_FEATURES):
            raise ValueError("Global feature definitions differ from implementation")
        means, stds = np.asarray(self.global_feature_means), np.asarray(self.global_feature_stds)
        if means.shape != (13,) or stds.shape != (13,) or not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
            raise ValueError("Invalid global normalization moments")
        if self.unknown_species_index != 0 or self.species_index_offset != 1:
            raise ValueError("Species index 0 must be reserved for unknown")
        if self.species_vocabulary != sorted(set(self.species_vocabulary)) or any(not x for x in self.species_vocabulary):
            raise ValueError("Species vocabulary must be sorted, unique, and nonempty")
        if not self.uses_species or self.uses_id_as_feature:
            raise ValueError("Invalid feature-use declarations")

    @property
    def num_species_embeddings(self) -> int:
        return len(self.species_vocabulary) + self.species_index_offset

    def species_index(self, species: str) -> int:
        # A dict is deliberately built only from label-free fitted metadata.
        return {name: i + self.species_index_offset for i, name in enumerate(self.species_vocabulary)}.get(
            species, self.unknown_species_index
        )

    def save(self, path: Union[str, Path]) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(text)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RepresentationMetadata":
        result = cls(**json.loads(Path(path).read_text()))
        result.validate()
        return result


def encode_sample(sequence: str, structure: str, mfe: float, species: str,
                  metadata: RepresentationMetadata,
                  reverse_complement: bool = False) -> dict[str, np.ndarray]:
    """Encode one raw sample in the ordinary or raw reverse-complement view.

    Reverse complementation happens before any feature computation.  Thus the
    structure channels, partner indices/channels, offsets, and all thirteen
    global features are derived afresh from the transformed raw strings.  The
    fixed training-fitted metadata is never refitted for the second view.
    """
    metadata.validate()
    if reverse_complement:
        sequence, structure, mfe = reverse_complement_view(sequence, structure, mfe)
    channels, partners = encode_representation(sequence, structure, metadata.length)
    raw_globals = raw_global_features(sequence, structure, float(mfe))
    globals_normalized = (
        (raw_globals - np.asarray(metadata.global_feature_means, dtype=np.float32)) /
        np.asarray(metadata.global_feature_stds, dtype=np.float32)
    ).astype(np.float32)
    result = {
        "channels": channels,
        "partner_indices": partners,
        "global_features": globals_normalized,
        "species_index": np.asarray(metadata.species_index(str(species)), dtype=np.int64),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Representation contains non-finite values")
    return result


class PreMiRNADataset(Dataset):
    """Lazy fixed-shape tensors; IDs are retained only in ``ids`` for output order."""
    def __init__(self, input_path: Union[str, Path], metadata: Union[RepresentationMetadata, str, Path],
                 labels_path: Optional[Union[str, Path]] = None,
                 reverse_complement: bool = False) -> None:
        self.metadata = metadata if isinstance(metadata, RepresentationMetadata) else RepresentationMetadata.load(metadata)
        self.metadata.validate()
        self.frame = read_samples(input_path)
        self.ids = self.frame["id"].astype(str).tolist()
        self.labels = read_labels(labels_path, self.ids, self.metadata.label_column) if labels_path is not None else None
        self.reverse_complement = bool(reverse_complement)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        encoded = encode_sample(
            row.sequence_rna, row.structure, row.mfe, row.species, self.metadata,
            reverse_complement=self.reverse_complement,
        )
        item = {key: torch.from_numpy(value) for key, value in encoded.items()}
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


class PairedViewDataset(Dataset):
    """Adapter exposing independently recomputed ordinary and RC tensor views.

    Each item is ``{"ordinary": ..., "reverse_complement": ...}``. Labels, when
    available, occur in both dictionaries and are joined by ID by the underlying
    datasets. This adapter is intended for tests/TTA; stochastic one-view training
    can use two :class:`PreMiRNADataset` instances with the same sample index.
    """
    def __init__(self, input_path: Union[str, Path], metadata: Union[RepresentationMetadata, str, Path],
                 labels_path: Optional[Union[str, Path]] = None) -> None:
        self.ordinary = PreMiRNADataset(input_path, metadata, labels_path, reverse_complement=False)
        self.reverse_complement = PreMiRNADataset(
            input_path, self.ordinary.metadata, labels_path, reverse_complement=True
        )
        self.ids = self.ordinary.ids
        if self.ids != self.reverse_complement.ids:
            raise RuntimeError("Paired views have different sample order")

    def __len__(self) -> int:
        return len(self.ordinary)

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "ordinary": self.ordinary[index],
            "reverse_complement": self.reverse_complement[index],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit training-only representation metadata")
    parser.add_argument("--train-input", required=True, help="Training input/ directory")
    parser.add_argument("--output", required=True, help="Output representation_metadata.json")
    args = parser.parse_args()
    metadata = RepresentationMetadata.fit(args.train_input)
    metadata.save(args.output)
    print(json.dumps({
        "rows": len(read_samples(args.train_input)), "species": len(metadata.species_vocabulary),
        "global_features": len(metadata.global_feature_names), "output": str(Path(args.output).resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
