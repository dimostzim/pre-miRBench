#!/usr/bin/env python3
"""
Build a prefixed multi-species training dataset.

Each species genome/BED is rewritten with species-prefixed contig names before
window extraction, so chromosome names are globally unique for MuStARD. Splits
are assigned globally by species and miRNA family so benchmark sets can separate
known-species/held-out-species and known-family/held-out-family generalization.
"""
import argparse
import csv
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "dataset"))
from prepare_tool_inputs import DEFAULT_TARGET_LENGTHS, crop_bounds as tool_crop_bounds, prepare_tool_inputs


TEST_SPLITS = (
    "test_known_species_known_family",
    "test_known_species_heldout_family",
    "test_heldout_species_known_family",
    "test_heldout_species_heldout_family",
)
FINAL_SPLITS = ("train", "valid", *TEST_SPLITS)
CANONICAL_LEAKAGE_LENGTH = 100
CANONICAL_LEAKAGE_FIELD = "canonical_100nt_sequence"
SPLIT_PRIORITY = {
    "test_heldout_species_heldout_family": 0,
    "test_heldout_species_known_family": 1,
    "test_known_species_heldout_family": 2,
    "test_known_species_known_family": 3,
    "valid": 4,
    "train": 5,
}

FIELDNAMES = [
    "record_id",
    "species",
    "split",
    "split_reason",
    "window_id",
    "chrom",
    "start",
    "end",
    "strand",
    "sequence",
    "structure",
    "mfe",
    "mirna_id",
    "family_id",
    "precursor_sequence",
    "canonical_100nt_sequence",
    "target_start",
    "target_end",
    "label",
    "score",
    "consensus",
    "hard_round",
]


def run(cmd, reuse_output=None):
    if reuse_output and Path(reuse_output).exists():
        print(f"reuse existing: {reuse_output}")
        return
    print("+", " ".join(str(part) for part in cmd))
    subprocess.check_call([str(part) for part in cmd])


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if str(path).endswith(".tsv") else ","))


def read_stats(path):
    if not Path(path).exists():
        return {}
    stats = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            stats[row["metric"]] = row["value"]
    return stats


def write_csv(path, rows, fieldnames=FIELDNAMES):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def find_single_file(directory, suffix):
    matches = sorted(Path(directory).glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No *{suffix} file found in {directory}")
    if len(matches) > 1:
        raise ValueError(f"Expected one *{suffix} file in {directory}, found: {matches}")
    return matches[0]


def load_panel(path, species_filter):
    rows = read_csv(path)
    if species_filter:
        keep = set(species_filter)
        rows = [row for row in rows if row["code"] in keep]
    output = []
    for row in rows:
        if row.get("status") != "auto":
            continue
        genome = row.get("genome")
        bed = row.get("bed")
        if not genome:
            genome = str(find_single_file(Path(path).parent / row["code"], ".fa"))
        if not bed:
            bed = str(Path(path).parent / row["code"] / f"{row['code']}-precursors-no-v2.bed")
        output.append(
            {
                "code": row["code"],
                "genome": genome,
                "bed": bed,
                "bed_rows": row.get("bed_rows", ""),
                "matched_rows": row.get("matched_rows", ""),
                "dropped_rows": row.get("dropped_rows", ""),
            }
        )
    if not output:
        raise SystemExit(f"No auto-download species found in {path}")
    return output


def prefixed_name(species, name):
    return f"{species}__{name}"


def write_prefixed_fasta(species, input_path, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(input_path) as input_handle, open(output_path, "w") as output_handle:
        for raw_line in input_handle:
            if raw_line.startswith(">"):
                header = raw_line[1:].strip()
                name, *rest = header.split(maxsplit=1)
                suffix = f" {rest[0]}" if rest else ""
                output_handle.write(f">{prefixed_name(species, name)}{suffix}\n")
            else:
                output_handle.write(raw_line)


def write_prefixed_bed(species, input_path, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(input_path) as input_handle, open(output_path, "w") as output_handle:
        for raw_line in input_handle:
            if not raw_line.strip() or raw_line.startswith("#"):
                output_handle.write(raw_line)
                continue
            parts = raw_line.rstrip("\n").split("\t")
            parts[0] = prefixed_name(species, parts[0])
            output_handle.write("\t".join(parts) + "\n")


def append_file(source, destination):
    with open(source) as input_handle, open(destination, "a") as output_handle:
        for line in input_handle:
            output_handle.write(line)


def parse_csv_set(value):
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def normalize_sequence(sequence):
    return str(sequence or "").strip().upper().replace("T", "U")


def family_id_from_mirna_id(mirna_id):
    name = str(mirna_id or "").replace("_pre", "")
    name = re.sub(r"^[A-Za-z]{3}-", "", name)
    if name.startswith("Let-7"):
        return "Let-7"

    mir_match = re.match(r"Mir-(\d+)", name)
    if mir_match:
        return f"Mir-{mir_match.group(1)}"

    numbered_match = re.match(r"([A-Za-z]+-\d+)", name)
    if numbered_match:
        return numbered_match.group(1)

    return re.sub(r"-(P\d+[A-Za-z]?\d*|v\d+).*$", "", name)


def precursor_sequence_for_row(row):
    if not row.get("target_start") or not row.get("target_end"):
        return ""

    sequence = normalize_sequence(row["sequence"])
    window_start = int(row["start"])
    window_end = int(row["end"])
    target_start = int(row["target_start"])
    target_end = int(row["target_end"])
    if row.get("strand") == "-":
        left = window_end - target_end
        right = window_end - target_start
    else:
        left = target_start - window_start
        right = target_end - window_start

    clipped_left = max(0, left)
    clipped_right = min(len(sequence), right)
    if clipped_left >= clipped_right:
        raise ValueError(
            f"Invalid precursor bounds for {row.get('mirna_id', row.get('window_id'))}: "
            f"window={window_start}-{window_end} target={target_start}-{target_end} "
            f"strand={row.get('strand')} sequence_length={len(sequence)}"
        )
    return sequence[clipped_left:clipped_right]


def canonical_leakage_sequence(row):
    if row.get(CANONICAL_LEAKAGE_FIELD):
        return normalize_sequence(row[CANONICAL_LEAKAGE_FIELD])
    if not row.get("sequence"):
        return normalize_sequence(row.get("precursor_sequence", ""))

    sequence = normalize_sequence(row["sequence"])
    left, right = tool_crop_bounds(row, len(sequence), CANONICAL_LEAKAGE_LENGTH)
    if row.get("strand") == "-":
        return sequence[len(sequence) - right:len(sequence) - left]
    return sequence[left:right]


def annotate_positive(row, species):
    row = row.copy()
    row["species"] = species
    row["family_id"] = family_id_from_mirna_id(row.get("mirna_id"))
    row["precursor_sequence"] = precursor_sequence_for_row(row)
    row[CANONICAL_LEAKAGE_FIELD] = canonical_leakage_sequence(row)
    row["split"] = ""
    row["split_reason"] = ""
    row["label"] = "1"
    return row


def annotate_negative(row, species):
    row = row.copy()
    row["species"] = species
    row["family_id"] = ""
    row["precursor_sequence"] = ""
    row[CANONICAL_LEAKAGE_FIELD] = canonical_leakage_sequence(row)
    row["split"] = ""
    row["split_reason"] = ""
    row["label"] = "0"
    return row


def grouped_by_family(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family_id"]].append(row)
    return grouped


def select_families_by_target(rows, target_rows, seed, excluded_families=None):
    if target_rows <= 0:
        return set()
    excluded_families = excluded_families or set()
    candidates = [
        (family_id, family_rows)
        for family_id, family_rows in grouped_by_family(rows).items()
        if family_id and family_id not in excluded_families
    ]
    random.Random(seed).shuffle(candidates)

    selected = set()
    selected_rows = 0
    for family_id, family_rows in candidates:
        if selected_rows >= target_rows:
            break
        selected.add(family_id)
        selected_rows += len(family_rows)
    return selected


def precursor_groups_by_family(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["family_id"]][row["precursor_sequence"]].append(row)
    return grouped


def select_precursor_groups_by_target(rows, target_rows, seed, assigned_groups=None):
    if target_rows <= 0:
        return set()
    assigned_groups = assigned_groups or set()
    grouped = precursor_groups_by_family(rows)
    remaining_counts = {
        family_id: sum(1 for precursor in groups if (family_id, precursor) not in assigned_groups)
        for family_id, groups in grouped.items()
    }
    candidates = []
    for family_id, groups in grouped.items():
        if remaining_counts[family_id] <= 1:
            continue
        for precursor, precursor_rows in groups.items():
            key = (family_id, precursor)
            if key not in assigned_groups:
                candidates.append((family_id, precursor, precursor_rows))

    random.Random(seed).shuffle(candidates)
    selected = set()
    selected_rows = 0
    for family_id, precursor, precursor_rows in candidates:
        if selected_rows >= target_rows:
            break
        if remaining_counts[family_id] <= 1:
            continue
        selected.add((family_id, precursor))
        remaining_counts[family_id] -= 1
        selected_rows += len(precursor_rows)
    return selected


def positive_priority_key(row):
    return (
        SPLIT_PRIORITY.get(row.get("split"), 999),
        row.get("species", ""),
        row.get("family_id", ""),
        row.get("mirna_id", ""),
        row.get("chrom", ""),
        int(row.get("start") or 0),
        row.get("window_id", ""),
    )


def deduplicate_positive_leakage_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[canonical_leakage_sequence(row)].append(row)

    kept = []
    excluded = []
    for sequence, sequence_rows in grouped.items():
        if not sequence:
            kept.extend(sequence_rows)
            continue
        ranked = sorted(sequence_rows, key=positive_priority_key)
        kept.append(ranked[0])
        for row in ranked[1:]:
            row = row.copy()
            row["split_reason"] = "duplicate_100nt_excluded"
            excluded.append(row)
    return kept, excluded


def relabel_family_splits(rows, heldout_species):
    rows = [row.copy() for row in rows]
    train_families = {row["family_id"] for row in rows if row["split"] == "train"}

    moved_valid = False
    for row in rows:
        if row["split"] == "valid" and row["family_id"] not in train_families:
            row["split"] = "train"
            row["split_reason"] = "valid_family_absent_from_train_moved_to_train"
            moved_valid = True

    if moved_valid:
        train_families = {row["family_id"] for row in rows if row["split"] == "train"}

    for row in rows:
        family_is_known = row["family_id"] in train_families
        if row["species"] in heldout_species:
            row["split"] = (
                "test_heldout_species_known_family"
                if family_is_known
                else "test_heldout_species_heldout_family"
            )
            row["split_reason"] = (
                "heldout_species_known_family"
                if family_is_known
                else "heldout_species_heldout_family"
            )
        elif row["split"].startswith("test_known_species"):
            row["split"] = (
                "test_known_species_known_family"
                if family_is_known
                else "test_known_species_heldout_family"
            )
            row["split_reason"] = (
                "known_species_known_family"
                if family_is_known
                else "known_species_heldout_family"
            )
    return rows


def assign_positive_splits(positives, args):
    heldout_species = parse_csv_set(args.heldout_species)
    known_species_rows = [row for row in positives if row["species"] not in heldout_species]
    heldout_species_families = {row["family_id"] for row in positives if row["species"] in heldout_species}

    valid_target = math.ceil(len(known_species_rows) * args.valid_frac)
    valid_heldout_family_target = math.ceil(valid_target * args.valid_heldout_family_frac)
    test_known_family_target = math.ceil(len(known_species_rows) * args.test_known_species_known_family_frac)
    test_heldout_family_target = math.ceil(len(known_species_rows) * args.test_known_species_heldout_family_frac)

    # Validation-only held-out families are kept out of final held-out species tests.
    valid_heldout_family_candidates = [
        row for row in known_species_rows if row["family_id"] not in heldout_species_families
    ]
    valid_heldout_families = select_families_by_target(
        valid_heldout_family_candidates,
        valid_heldout_family_target,
        args.seed + 101,
    )
    test_heldout_families = select_families_by_target(
        known_species_rows,
        test_heldout_family_target,
        args.seed + 202,
        excluded_families=valid_heldout_families,
    )

    train_candidate_rows = [
        row
        for row in known_species_rows
        if row["family_id"] not in valid_heldout_families
        and row["family_id"] not in test_heldout_families
    ]
    if known_species_rows and not train_candidate_rows:
        family_counts = Counter(row["family_id"] for row in known_species_rows)
        family_to_keep, _count = family_counts.most_common(1)[0]
        valid_heldout_families.discard(family_to_keep)
        test_heldout_families.discard(family_to_keep)
        train_candidate_rows = [
            row
            for row in known_species_rows
            if row["family_id"] not in valid_heldout_families
            and row["family_id"] not in test_heldout_families
        ]
    valid_known_target = max(0, valid_target - sum(1 for row in known_species_rows if row["family_id"] in valid_heldout_families))
    valid_known_groups = select_precursor_groups_by_target(
        train_candidate_rows,
        valid_known_target,
        args.seed + 303,
    )
    test_known_groups = select_precursor_groups_by_target(
        train_candidate_rows,
        test_known_family_target,
        args.seed + 404,
        assigned_groups=valid_known_groups,
    )

    assigned = []
    for row in positives:
        row = row.copy()
        family_id = row["family_id"]
        precursor_key = (family_id, row["precursor_sequence"])
        if row["species"] in heldout_species:
            row["split"] = ""
            row["split_reason"] = "heldout_species_pending"
        elif family_id in valid_heldout_families:
            row["split"] = "valid"
            row["split_reason"] = "known_species_heldout_family"
        elif family_id in test_heldout_families:
            row["split"] = "test_known_species_heldout_family"
            row["split_reason"] = "known_species_heldout_family"
        elif precursor_key in valid_known_groups:
            row["split"] = "valid"
            row["split_reason"] = "known_species_known_family"
        elif precursor_key in test_known_groups:
            row["split"] = "test_known_species_known_family"
            row["split_reason"] = "known_species_known_family"
        else:
            row["split"] = "train"
            row["split_reason"] = "train"
        assigned.append(row)

    train_families = {row["family_id"] for row in assigned if row["split"] == "train"}
    for row in assigned:
        if row["split"]:
            continue
        if row["family_id"] in train_families:
            row["split"] = "test_heldout_species_known_family"
            row["split_reason"] = "heldout_species_known_family"
        else:
            row["split"] = "test_heldout_species_heldout_family"
            row["split_reason"] = "heldout_species_heldout_family"

    kept, excluded = deduplicate_positive_leakage_groups(assigned)
    kept = relabel_family_splits(kept, heldout_species)
    return kept, excluded


def negative_candidate_order(species_result):
    seen_windows = set()
    ordered = []
    hard_sorted = sorted(
        species_result["hard_negatives"],
        key=lambda row: (
            int(row.get("hard_round") or 999999),
            -float(row.get("score") or 0.0),
        ),
    )
    scored_sorted = sorted(
        species_result["scored_negatives"],
        key=lambda row: float(row.get("score") or 0.0),
        reverse=True,
    )
    for row in [*hard_sorted, *scored_sorted]:
        if row["window_id"] in seen_windows:
            continue
        seen_windows.add(row["window_id"])
        ordered.append(row)
    return ordered


def split_species_quotas(positive_rows, ratio):
    counts = Counter(row["species"] for row in positive_rows)
    target = math.ceil(len(positive_rows) * ratio)
    quotas = {species: math.floor(count * ratio) for species, count in counts.items()}
    remainder = target - sum(quotas.values())
    fractions = sorted(
        ((count * ratio) - math.floor(count * ratio), species)
        for species, count in counts.items()
    )
    for _fraction, species in reversed(fractions):
        if remainder <= 0:
            break
        quotas[species] += 1
        remainder -= 1
    return quotas, target


def take_negative_rows(species, candidates_by_species, needed, used_windows, used_sequences):
    chosen = []
    for row in candidates_by_species.get(species, []):
        if len(chosen) >= needed:
            break
        sequence_key = canonical_leakage_sequence(row)
        if not sequence_key or sequence_key in used_sequences or row["window_id"] in used_windows:
            continue
        used_sequences.add(sequence_key)
        used_windows.add(row["window_id"])
        chosen.append(row)
    return chosen


def select_all_negatives(results, positives_by_split, args):
    result_by_species = {result["species"]: result for result in results}
    candidates_by_species = {
        species: negative_candidate_order(result)
        for species, result in result_by_species.items()
    }
    used_windows = set()
    used_sequences = {
        canonical_leakage_sequence(row)
        for split_rows in positives_by_split.values()
        for row in split_rows
        if canonical_leakage_sequence(row)
    }
    selected = []
    issues_by_species = defaultdict(list)

    for split_index, split in enumerate(FINAL_SPLITS):
        positive_rows = positives_by_split.get(split, [])
        if not positive_rows:
            continue

        quotas, target = split_species_quotas(positive_rows, args.ratio)
        split_selected = []
        for species, quota in sorted(quotas.items(), key=lambda item: result_by_species[item[0]]["species_index"]):
            split_selected.extend(
                take_negative_rows(
                    species,
                    candidates_by_species,
                    quota,
                    used_windows,
                    used_sequences,
                )
            )

        if len(split_selected) < target:
            rng = random.Random(args.seed + split_index)
            fill_species = list(quotas)
            rng.shuffle(fill_species)
            while len(split_selected) < target:
                before = len(split_selected)
                for species in fill_species:
                    if len(split_selected) >= target:
                        break
                    split_selected.extend(
                        take_negative_rows(
                            species,
                            candidates_by_species,
                            1,
                            used_windows,
                            used_sequences,
                        )
                    )
                if len(split_selected) == before:
                    break

        for row in split_selected:
            row = row.copy()
            row["split"] = split
            row["split_reason"] = "negative"
            selected.append(row)

        if len(split_selected) < target:
            message = (
                f"{split}: requested {target} negatives for "
                f"{len(positive_rows)} positives but selected {len(split_selected)}"
            )
            for species in quotas:
                issues_by_species[species].append(message)

    return selected, issues_by_species


def write_split_summary(path, rows):
    fieldnames = [
        "species",
        "is_heldout_species",
        "bed_rows",
        "bed_matched_rows",
        "bed_dropped_rows",
        "bed_pre_entries",
        "positives_after_filters",
        "positives_written",
        "positive_chr_missing",
        "positive_boundary_filtered",
        "positive_repeat_filtered",
        "negative_windows_folded",
        "negative_hairpin_like_kept",
        "negative_overlap_skipped",
        "negative_repeat_or_n_skipped",
        "input_positives",
        "input_negative_windows",
        "excluded_pos",
        "positives",
        "negatives",
        *[f"{split}_pos" for split in FINAL_SPLITS],
        *[f"{split}_neg" for split in FINAL_SPLITS],
        "issues",
    ]
    write_csv(path, rows, fieldnames=fieldnames)


def split_positive_counts(rows):
    return Counter((row["species"], row["split"]) for row in rows if row["label"] == "1")


def finalize_records(rows):
    counters = Counter()
    finalized = []
    for row in rows:
        row = row.copy()
        label = "1" if str(row.get("label")) == "1" else "0"
        counters[(row["species"], label)] += 1
        stem = "pos" if label == "1" else "neg"
        row["record_id"] = f"{row['species']}_{stem}_{counters[(row['species'], label)]:06d}"
        row["label"] = label
        finalized.append(row)
    return finalized


def write_family_split_summary(path, rows, heldout_species):
    train_families = {row["family_id"] for row in rows if row["split"] == "train" and row["label"] == "1"}
    train_sequences = {
        canonical_leakage_sequence(row)
        for row in rows
        if row["split"] == "train" and row["label"] == "1" and canonical_leakage_sequence(row)
    }

    summary_rows = []
    for split in FINAL_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        positive_rows = [row for row in split_rows if row["label"] == "1"]
        negative_rows = [row for row in split_rows if row["label"] == "0"]
        summary_rows.append(
            {
                "split": split,
                "positives": len(positive_rows),
                "negatives": len(negative_rows),
                "species_count": len({row["species"] for row in positive_rows}),
                "family_count": len({row["family_id"] for row in positive_rows}),
                "known_species_positives": sum(1 for row in positive_rows if row["species"] not in heldout_species),
                "heldout_species_positives": sum(1 for row in positive_rows if row["species"] in heldout_species),
                "known_family_positives": sum(1 for row in positive_rows if row["family_id"] in train_families),
                "heldout_family_positives": sum(1 for row in positive_rows if row["family_id"] not in train_families),
                "exact_100nt_overlap_with_train": (
                    0
                    if split == "train"
                    else sum(1 for row in positive_rows if canonical_leakage_sequence(row) in train_sequences)
                ),
            }
        )

    fieldnames = [
        "split",
        "positives",
        "negatives",
        "species_count",
        "family_count",
        "known_species_positives",
        "heldout_species_positives",
        "known_family_positives",
        "heldout_family_positives",
        "exact_100nt_overlap_with_train",
    ]
    write_csv(path, summary_rows, fieldnames=fieldnames)


def validate_split_guarantees(rows, heldout_species):
    train_positive_rows = [row for row in rows if row["split"] == "train" and row["label"] == "1"]
    train_families = {row["family_id"] for row in train_positive_rows}
    seen_sequences = {}
    issues = []

    for row in rows:
        sequence_key = canonical_leakage_sequence(row)
        if not sequence_key:
            issues.append(f"{row.get('record_id')}: missing canonical 100nt sequence")
        elif sequence_key in seen_sequences:
            other = seen_sequences[sequence_key]
            issues.append(
                f"{row.get('record_id')}: duplicate canonical 100nt sequence with "
                f"{other.get('record_id')} ({other.get('split')})"
            )
        else:
            seen_sequences[sequence_key] = row

        if row["label"] != "1":
            continue
        split = row["split"]
        if split.startswith("test_known_species") and row["species"] in heldout_species:
            issues.append(f"{row.get('mirna_id', row.get('record_id'))}: held-out species in known-species split")
        if split.startswith("test_heldout_species") and row["species"] not in heldout_species:
            issues.append(f"{row.get('mirna_id', row.get('record_id'))}: known species in held-out-species split")
        if split.endswith("heldout_family") and row["family_id"] in train_families:
            issues.append(f"{row.get('mirna_id', row.get('record_id'))}: train family in held-out-family split")
        if split.endswith("known_family") and row["family_id"] not in train_families:
            issues.append(f"{row.get('mirna_id', row.get('record_id'))}: held-out family in known-family split")

    if issues:
        examples = "\n".join(f"  - {issue}" for issue in issues[:20])
        suffix = f"\n  ... {len(issues) - 20} more" if len(issues) > 20 else ""
        raise ValueError(f"Split guarantee validation failed:\n{examples}{suffix}")


def validate_ratio_guarantees(rows, ratio):
    issues = []
    counts = Counter((row["split"], row["label"]) for row in rows)
    for split in FINAL_SPLITS:
        positives = counts[(split, "1")]
        negatives = counts[(split, "0")]
        target = math.ceil(positives * ratio)
        if negatives != target:
            issues.append(f"{split}: positives={positives} negatives={negatives} expected_negatives={target}")
    if issues:
        raise ValueError("Negative ratio validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues))


def write_leakage_report(path, rows, excluded_positive_rows, ratio):
    counts = Counter((row["split"], row["label"]) for row in rows)
    report_rows = []
    report_rows.append({"metric": "canonical_leakage_length", "value": CANONICAL_LEAKAGE_LENGTH})
    report_rows.append({"metric": "records", "value": len(rows)})
    report_rows.append({"metric": "unique_100nt_sequences", "value": len({canonical_leakage_sequence(row) for row in rows})})
    report_rows.append({"metric": "excluded_positive_100nt_duplicates", "value": len(excluded_positive_rows)})
    for split in FINAL_SPLITS:
        positives = counts[(split, "1")]
        negatives = counts[(split, "0")]
        report_rows.append({"metric": f"{split}_positives", "value": positives})
        report_rows.append({"metric": f"{split}_negatives", "value": negatives})
        report_rows.append({"metric": f"{split}_expected_negatives", "value": math.ceil(positives * ratio)})
    write_csv(path, report_rows, fieldnames=["metric", "value"])


def split_positive_rows(results, args):
    positives = []
    for result in results:
        positives.extend(result["positives"])
    return assign_positive_splits(positives, args)


def assemble_dataset_rows(results, args, positive_rows, excluded_positive_rows):
    positives_by_split = defaultdict(list)
    for row in positive_rows:
        positives_by_split[row["split"]].append(row)

    negatives, issues_by_species = select_all_negatives(results, positives_by_split, args)

    rows = []
    result_by_species = {result["species"]: result for result in results}
    split_order = {split: index for index, split in enumerate(FINAL_SPLITS)}
    for species in sorted(result_by_species, key=lambda value: result_by_species[value]["species_index"]):
        species_positives = [row for row in positive_rows if row["species"] == species]
        species_negatives = [row for row in negatives if row["species"] == species]
        species_rows = species_positives + species_negatives
        rows.extend(
            sorted(
                species_rows,
                key=lambda row: (
                    split_order.get(row["split"], 999),
                    row["label"],
                    row.get("chrom", ""),
                    int(row.get("start") or 0),
                    row.get("mirna_id", ""),
                    row.get("window_id", ""),
                ),
            )
        )

    final_rows = finalize_records(rows)
    validate_split_guarantees(final_rows, parse_csv_set(args.heldout_species))
    validate_ratio_guarantees(final_rows, args.ratio)
    counts = Counter((row["species"], row["split"], row["label"]) for row in final_rows)
    excluded_counts = Counter(row["species"] for row in excluded_positive_rows)
    for result in results:
        species = result["species"]
        summary = result["summary"]
        summary["excluded_pos"] = excluded_counts[species]
        summary["positives"] = sum(counts[(species, split, "1")] for split in FINAL_SPLITS)
        summary["negatives"] = sum(counts[(species, split, "0")] for split in FINAL_SPLITS)
        for split in FINAL_SPLITS:
            summary[f"{split}_pos"] = counts[(species, split, "1")]
            summary[f"{split}_neg"] = counts[(species, split, "0")]
        issue_parts = []
        if excluded_counts[species]:
            issue_parts.append(f"excluded {excluded_counts[species]} duplicate 100nt positives")
        issue_parts.extend(issues_by_species[species])
        summary["issues"] = "; ".join(issue_parts)

    return final_rows, excluded_positive_rows


def process_species(species_index, species_count, species_row, args, script_dir, python_exe, mining_jobs):
    species = species_row["code"]
    print(f"\n### species {species} ({species_index}/{species_count})")
    work_dir = Path(args.work_dir)
    species_work = work_dir / species
    species_work.mkdir(parents=True, exist_ok=True)
    prefixed_genome = species_work / "genome.prefixed.fa"
    prefixed_bed = species_work / "precursors.prefixed.bed"
    positives_csv = species_work / "positives.csv"
    positives_stats_csv = species_work / "positives.stats.csv"
    pool_csv = species_work / "hairpin_pool.csv"
    pool_stats_csv = species_work / "hairpin_pool.stats.csv"
    hard_csv = species_work / "hard_negatives.train_mined.csv"
    scores_csv = species_work / "hard_negative_scores.train_mined.csv"

    if not args.reuse_existing or not prefixed_genome.exists():
        write_prefixed_fasta(species, species_row["genome"], prefixed_genome)
    if not args.reuse_existing or not prefixed_bed.exists():
        write_prefixed_bed(species, species_row["bed"], prefixed_bed)

    run(
        [
            python_exe,
            script_dir / "validate_bed_genome.py",
            "--bed",
            prefixed_bed,
            "--genome",
            prefixed_genome,
        ],
        None,
    )
    run(
        [
            python_exe,
            script_dir / "extract_positives.py",
            "--bed",
            prefixed_bed,
            "--genome",
            prefixed_genome,
            "--window",
            args.window,
            "--max-repeat-frac",
            args.max_repeat_frac,
            "--output",
            positives_csv,
            "--stats-output",
            positives_stats_csv,
            "--cpus",
            args.cpus,
        ],
        positives_csv if args.reuse_existing else None,
    )

    hairpin_cmd = [
        python_exe,
        script_dir / "extract_hairpins.py",
        "--bed",
        prefixed_bed,
        "--genome",
        prefixed_genome,
        "--window",
        args.window,
        "--step",
        args.step,
        "--max-repeat-frac",
        args.max_repeat_frac,
        "--min-mfe",
        args.min_mfe,
        "--min-paired-frac",
        args.min_paired_frac,
        "--min-stem",
        args.min_stem,
        "--max-loop",
        args.max_loop,
        "--output",
        pool_csv,
        "--stats-output",
        pool_stats_csv,
        "--cpus",
        args.cpus,
    ]
    if args.max_negative_windows_per_species:
        hairpin_cmd.extend(["--max-windows", args.max_negative_windows_per_species])
    if not args.sequential_negative_scan:
        hairpin_cmd.append("--balance-bed-chroms")
    run(hairpin_cmd, pool_csv if args.reuse_existing else None)

    heldout_species = {item.strip() for item in args.heldout_species.split(",") if item.strip()}
    positives_stats = read_stats(positives_stats_csv)
    pool_stats = read_stats(pool_stats_csv)
    positives = [annotate_positive(row, species) for row in read_csv(positives_csv)]

    summary_row = {
        "species": species,
        "is_heldout_species": int(species in heldout_species),
        "bed_rows": species_row.get("bed_rows", ""),
        "bed_matched_rows": species_row.get("matched_rows", ""),
        "bed_dropped_rows": species_row.get("dropped_rows", ""),
        "bed_pre_entries": positives_stats.get("bed_pre_entries", ""),
        "positives_after_filters": positives_stats.get("positives_after_filters", ""),
        "positives_written": positives_stats.get("positives_written", ""),
        "positive_chr_missing": positives_stats.get("skipped_chr_missing", ""),
        "positive_boundary_filtered": positives_stats.get("skipped_boundary", ""),
        "positive_repeat_filtered": positives_stats.get("skipped_repeat", ""),
        "negative_windows_folded": pool_stats.get("folded_candidate_windows", ""),
        "negative_hairpin_like_kept": pool_stats.get("hairpin_like_negatives_kept", ""),
        "negative_overlap_skipped": pool_stats.get("skipped_overlap", ""),
        "negative_repeat_or_n_skipped": pool_stats.get("skipped_repeat_or_n", ""),
        "input_positives": len(positives),
        "input_negative_windows": pool_stats.get("hairpin_like_negatives_kept", ""),
        "excluded_pos": 0,
        "positives": 0,
        "negatives": 0,
        "issues": "",
    }
    return {
        "species_index": species_index,
        "species": species,
        "prefixed_genome": str(prefixed_genome),
        "pool_csv": str(pool_csv),
        "hard_csv": str(hard_csv),
        "scores_csv": str(scores_csv),
        "positives": positives,
        "hard_negatives": [],
        "scored_negatives": [],
        "summary": summary_row,
    }


def mine_species_negatives(species_result, mining_positives, args, script_dir, python_exe, mining_jobs):
    species = species_result["species"]
    species_work = Path(args.work_dir) / species
    mining_positives_csv = species_work / "mining_train_positives.csv"
    write_csv(mining_positives_csv, mining_positives)

    hard_csv = Path(species_result["hard_csv"])
    scores_csv = Path(species_result["scores_csv"])
    run(
        [
            python_exe,
            script_dir / "mine_negatives.py",
            "--positives",
            mining_positives_csv,
            "--pool",
            species_result["pool_csv"],
            "--hard-negatives",
            hard_csv,
            "--scores",
            scores_csv,
            "--ratio",
            args.ratio,
            "--rounds",
            args.mining_rounds,
            "--ensemble-size",
            args.ensemble_size,
            "--trees",
            args.trees,
            "--consensus",
            args.consensus,
            "--jobs",
            mining_jobs,
            "--seed",
            args.seed + species_result["species_index"],
        ],
        scores_csv if args.reuse_existing else None,
    )
    return {
        "species": species,
        "hard_negatives": [annotate_negative(row, species) for row in read_csv(hard_csv)],
        "scored_negatives": [annotate_negative(row, species) for row in read_csv(scores_csv)],
    }


def add_train_mined_negatives(results, positive_rows, args, script_dir, python_exe, mining_jobs):
    train_positives = [row for row in positive_rows if row["split"] == "train"]
    if not train_positives:
        raise ValueError("No train positives available for negative mining.")

    train_by_species = defaultdict(list)
    for row in train_positives:
        train_by_species[row["species"]].append(row)

    updates = []
    species_jobs = max(1, args.species_jobs)
    if species_jobs == 1:
        for result in sorted(results, key=lambda row: row["species_index"]):
            mining_positives = train_by_species.get(result["species"]) or train_positives
            updates.append(
                mine_species_negatives(
                    result,
                    mining_positives,
                    args,
                    script_dir,
                    python_exe,
                    mining_jobs,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=species_jobs) as executor:
            futures = {}
            for result in results:
                mining_positives = train_by_species.get(result["species"]) or train_positives
                futures[
                    executor.submit(
                        mine_species_negatives,
                        result,
                        mining_positives,
                        args,
                        script_dir,
                        python_exe,
                        mining_jobs,
                    )
                ] = result["species"]
            for future in as_completed(futures):
                species = futures[future]
                updates.append(future.result())
                print(f"### mined negatives {species}")

    update_by_species = {update["species"]: update for update in updates}
    for result in results:
        update = update_by_species[result["species"]]
        result["hard_negatives"] = update["hard_negatives"]
        result["scored_negatives"] = update["scored_negatives"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a prefixed multi-species 1:N training dataset.")
    parser.add_argument("--panel", default="data/raw/mirgenedb_71/panel.tsv")
    parser.add_argument("--output-dir", default="data/datasets/mirgenedb_71")
    parser.add_argument("--work-dir", default="data/work/build_mirgenedb_71")
    parser.add_argument("--species", default=None, help="Comma-separated species codes to include. Default: all auto species in panel.")
    parser.add_argument("--heldout-species", default="gga,dme")
    parser.add_argument("--valid-frac", type=float, default=0.10)
    parser.add_argument("--valid-heldout-family-frac", type=float, default=0.0)
    parser.add_argument("--test-known-species-known-family-frac", type=float, default=0.10)
    parser.add_argument("--test-known-species-heldout-family-frac", type=float, default=0.10)
    parser.add_argument("--ratio", type=float, default=5.0)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--max-negative-windows-per-species", type=int, default=50000)
    parser.add_argument(
        "--sequential-negative-scan",
        action="store_true",
        help="Scan negative windows in FASTA order. Default balances capped scans across BED-positive chromosomes.",
    )
    parser.add_argument("--max-repeat-frac", type=float, default=0.1)
    parser.add_argument("--min-mfe", type=float, default=-10.0)
    parser.add_argument("--min-paired-frac", type=float, default=0.40)
    parser.add_argument("--min-stem", type=int, default=8)
    parser.add_argument("--max-loop", type=int, default=25)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--species-jobs", type=int, default=1, help="Number of species to process in parallel.")
    parser.add_argument("--mining-rounds", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=10)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--consensus", type=float, default=0.5)
    parser.add_argument(
        "--mining-jobs",
        type=int,
        default=None,
        help="RandomForest jobs per species. Default: -1 sequentially, 1 when --species-jobs > 1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ratio <= 0:
        raise SystemExit("--ratio must be positive")
    for name in (
        "valid_frac",
        "valid_heldout_family_frac",
        "test_known_species_known_family_frac",
        "test_known_species_heldout_family_frac",
    ):
        value = getattr(args, name)
        if value < 0 or value > 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")

    script_dir = Path(__file__).resolve().parent / "dataset"
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    species_filter = [item.strip() for item in args.species.split(",") if item.strip()] if args.species else None
    panel_rows = load_panel(args.panel, species_filter)

    combined_genome = output_dir / "genome.fa"
    if combined_genome.exists():
        combined_genome.unlink()

    summary_rows = []
    python_exe = sys.executable
    species_jobs = max(1, args.species_jobs)
    mining_jobs = args.mining_jobs
    if mining_jobs is None:
        mining_jobs = 1 if species_jobs > 1 else -1
    print(f"species jobs: {species_jobs}")
    print(f"RNAfold jobs per species: {args.cpus}")
    print(f"RandomForest jobs per species: {mining_jobs}")

    results = []
    if species_jobs == 1:
        for species_index, species_row in enumerate(panel_rows, start=1):
            results.append(
                process_species(
                    species_index,
                    len(panel_rows),
                    species_row,
                    args,
                    script_dir,
                    python_exe,
                    mining_jobs,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=species_jobs) as executor:
            futures = {
                executor.submit(
                    process_species,
                    species_index,
                    len(panel_rows),
                    species_row,
                    args,
                    script_dir,
                    python_exe,
                    mining_jobs,
                ): species_row["code"]
                for species_index, species_row in enumerate(panel_rows, start=1)
            }
            for future in as_completed(futures):
                species = futures[future]
                result = future.result()
                print(f"### completed {species}")
                results.append(result)

    for result in sorted(results, key=lambda row: row["species_index"]):
        append_file(result["prefixed_genome"], combined_genome)
        summary_rows.append(result["summary"])

    positive_rows, excluded_positive_rows = split_positive_rows(results, args)
    add_train_mined_negatives(results, positive_rows, args, script_dir, python_exe, mining_jobs)
    all_rows, excluded_positive_rows = assemble_dataset_rows(results, args, positive_rows, excluded_positive_rows)
    heldout_species = parse_csv_set(args.heldout_species)
    dataset_csv = output_dir / "dataset.csv"
    split_summary = output_dir / "split_summary.csv"
    family_split_summary = output_dir / "family_split_summary.csv"
    leakage_report = output_dir / "leakage_report.csv"
    write_csv(dataset_csv, all_rows)
    write_split_summary(split_summary, summary_rows)
    write_family_split_summary(family_split_summary, all_rows, heldout_species)
    write_leakage_report(leakage_report, all_rows, excluded_positive_rows, args.ratio)
    prepare_tool_inputs(all_rows, output_dir / "tool_inputs", target_lengths=DEFAULT_TARGET_LENGTHS)

    print("\nsummary")
    print(f"species: {len(panel_rows)}")
    print(f"records: {len(all_rows)}")
    print(f"excluded positives: {len(excluded_positive_rows)}")
    print(f"dataset: {dataset_csv}")
    print(f"split summary: {split_summary}")
    print(f"family split summary: {family_split_summary}")
    print(f"leakage report: {leakage_report}")
    print(f"combined genome: {combined_genome}")
    print(f"tool inputs: {output_dir / 'tool_inputs'}")
    issue_rows = [row for row in summary_rows if row.get("issues")]
    if issue_rows:
        print("\nissues")
        for row in issue_rows:
            print(f"{row['species']}: {row['issues']}")
    else:
        print("issues: none")


if __name__ == "__main__":
    main()
