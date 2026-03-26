#!/usr/bin/env python3
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

TOOLS = ("deepmir", "deepmirgene", "dnnpremir", "mirdnn", "mire2e", "mustard")
FIXED_LENGTHS = {
    "dnnpremir": 180,
    "mirdnn": 160,
    "mire2e": 100,
    "mustard": 100,
}
RESULT_FILES = {
    "deepmir": "results.csv",
    "deepmirgene": "predictions.txt",
    "dnnpremir": "predictions.txt",
    "mirdnn": "predictions.csv",
    "mire2e": "predictions.json",
}


def parse_tools(value):
    if not value or value == "all":
        return list(TOOLS)
    tools = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(tools) - set(TOOLS))
    if unknown:
        raise ValueError(f"Unsupported tools: {', '.join(unknown)}")
    return tools


def normalize_sequence(sequence):
    return sequence.strip().upper().replace("T", "U")


def load_target_mirna_coords(path):
    coords = {}
    with open(path) as handle:
        for raw_line in handle:
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            chrom = parts[0]
            start = int(parts[1]) + 1
            end = int(parts[2])
            name = parts[3]
            coords[name] = {
                "chrom": chrom,
                "start": start,
                "end": end,
            }
    return coords


def center_crop_sequence(sequence, target_length):
    if target_length is None or len(sequence) <= target_length:
        return sequence, 0
    trim = len(sequence) - target_length
    left_trim = trim // 2
    right_trim = trim - left_trim
    return sequence[left_trim:len(sequence) - right_trim], left_trim


def center_resize_interval(start, end, target_length):
    current_length = end - start + 1
    if current_length == target_length:
        return start, end

    if current_length > target_length:
        trim = current_length - target_length
        left_trim = trim // 2
        right_trim = trim - left_trim
        return start + left_trim, end - right_trim

    expand = target_length - current_length
    left_expand = expand // 2
    right_expand = expand - left_expand
    new_start = max(1, start - left_expand)
    new_end = end + right_expand
    if new_end - new_start + 1 < target_length:
        new_end += target_length - (new_end - new_start + 1)
    return new_start, new_end


def target_aware_resize_interval(start, end, target_start, target_end, target_length):
    current_length = end - start + 1
    if current_length <= target_length:
        return start, end

    max_start = end - target_length + 1
    target_mid = (target_start + target_end) / 2.0
    proposed_start = int(round(target_mid - ((target_length - 1) / 2.0)))
    new_start = max(start, min(proposed_start, max_start))
    new_end = new_start + target_length - 1

    if not (new_start <= target_start and new_end >= target_end):
        leftmost_start = max(start, target_end - target_length + 1)
        rightmost_start = min(max_start, target_start)
        if leftmost_start > rightmost_start:
            raise ValueError(
                f"Cannot fit target interval {target_start}-{target_end} into {target_length} nt "
                f"within source interval {start}-{end}"
            )
        new_start = min(max(new_start, leftmost_start), rightmost_start)
        new_end = new_start + target_length - 1

    return new_start, new_end


def build_source_records(rows, prefix, target_mirna_coords=None):
    records = []
    for row_index, row in enumerate(rows, start=1):
        sequence = normalize_sequence(row["sequence"])
        target_info = {}
        target_name = row.get("target_mirna", "")
        if target_mirna_coords and target_name in target_mirna_coords:
            target_info = target_mirna_coords[target_name]
        record = {
            "record_id": f"{prefix}__{row_index:06d}",
            "window_id": row["window_id"],
            "chrom": row["chrom"],
            "start": int(row["start"]),
            "end": int(row["end"]),
            "strand": row.get("strand", "."),
            "label": row.get("label", ""),
            "sequence": sequence,
            "sequence_length": len(sequence),
            "target_mirna": target_name,
            "contained_mirnas": row.get("contained_mirnas", ""),
            "target_start": target_info.get("start"),
            "target_end": target_info.get("end"),
        }
        records.append(record)
    return records


def prepare_record_for_tool(tool, source_record):
    target_length = FIXED_LENGTHS.get(tool)
    if target_length is None:
        prepared_start = source_record["start"]
        prepared_end = source_record["end"]
    elif source_record.get("label") == "positive" and source_record.get("target_start") and source_record.get("target_end"):
        prepared_start, prepared_end = target_aware_resize_interval(
            source_record["start"],
            source_record["end"],
            source_record["target_start"],
            source_record["target_end"],
            target_length,
        )
    else:
        prepared_start, prepared_end = center_resize_interval(
            source_record["start"],
            source_record["end"],
            target_length,
        )

    left_trim = prepared_start - source_record["start"]
    prepared_sequence = source_record["sequence"][left_trim:left_trim + (prepared_end - prepared_start + 1)]

    return {
        "record_id": source_record["record_id"],
        "window_id": source_record["window_id"],
        "chrom": source_record["chrom"],
        "original_start": source_record["start"],
        "original_end": source_record["end"],
        "prepared_start": prepared_start,
        "prepared_end": prepared_end,
        "strand": source_record["strand"],
        "label": source_record["label"],
        "original_sequence_length": source_record["sequence_length"],
        "prepared_sequence_length": len(prepared_sequence),
        "sequence": prepared_sequence,
        "target_mirna": source_record.get("target_mirna", ""),
        "contained_mirnas": source_record.get("contained_mirnas", ""),
        "target_start": source_record.get("target_start"),
        "target_end": source_record.get("target_end"),
        "left_trim": left_trim,
    }


def write_source_manifest(records, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{prefix}.source.csv"
    fieldnames = [
        "record_id",
        "window_id",
        "chrom",
        "start",
        "end",
        "strand",
        "label",
        "sequence_length",
        "target_mirna",
        "contained_mirnas",
        "target_start",
        "target_end",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "record_id": record["record_id"],
                "window_id": record["window_id"],
                "chrom": record["chrom"],
                "start": record["start"],
                "end": record["end"],
                "strand": record["strand"],
                "label": record["label"],
                "sequence_length": record["sequence_length"],
                "target_mirna": record["target_mirna"],
                "contained_mirnas": record["contained_mirnas"],
                "target_start": record["target_start"] or "",
                "target_end": record["target_end"] or "",
            })
    return manifest_path


def write_tool_inputs(tool, source_records, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_records = [prepare_record_for_tool(tool, record) for record in source_records]

    fasta_path = output_dir / f"{prefix}.fa"
    bed_path = output_dir / f"{prefix}.bed"
    metadata_path = output_dir / f"{prefix}.metadata.csv"

    with fasta_path.open("w") as fasta_out, bed_path.open("w", newline="") as bed_out, metadata_path.open("w", newline="") as meta_out:
        bed_writer = csv.writer(bed_out, delimiter="\t")
        meta_fieldnames = [
            "record_id",
            "window_id",
            "chrom",
            "original_start",
            "original_end",
            "prepared_start",
            "prepared_end",
            "strand",
            "label",
            "original_sequence_length",
            "prepared_sequence_length",
            "target_mirna",
            "contained_mirnas",
            "target_start",
            "target_end",
        ]
        meta_writer = csv.DictWriter(meta_out, fieldnames=meta_fieldnames)
        meta_writer.writeheader()

        for record in prepared_records:
            fasta_out.write(f">{record['record_id']}\n{record['sequence']}\n")
            bed_writer.writerow([
                record["chrom"],
                record["prepared_start"] - 1,
                record["prepared_end"],
                record["record_id"],
                0,
                record["strand"],
            ])
            meta_writer.writerow({
                key: record[key]
                for key in meta_fieldnames
            })

    return {
        "fasta": fasta_path,
        "bed": bed_path,
        "metadata": metadata_path,
        "records": prepared_records,
    }


def load_metadata(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_record = {row["record_id"]: row for row in rows}
    return rows, by_record


def gt_to_int(label):
    return 1 if label == "positive" else 0


def _parse_optional_score(value):
    if value in ("", None):
        return None
    return float(value)


def normalize_deepmir(output_path, metadata_by_record):
    rows = []
    with open(output_path, newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = row["hairpin"]
            predicted_class = 1 if row["label"] == "pre-miRNA" else 0
            rows.append({
                "record_id": record_id,
                "window_id": metadata_by_record[record_id]["window_id"],
                "score": None,
                "predicted_class": predicted_class,
                "ground_truth_class": gt_to_int(metadata_by_record[record_id]["label"]),
            })
    return rows


def normalize_deepmirgene(output_path, metadata_rows):
    preds = []
    with open(output_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                preds.append(line.split()[-1])

    if len(preds) != len(metadata_rows):
        raise RuntimeError(f"deepmirgene count mismatch: {len(preds)} predictions vs {len(metadata_rows)} metadata rows")

    rows = []
    for meta, pred in zip(metadata_rows, preds):
        predicted_class = 1 if pred == "0" else 0
        rows.append({
            "record_id": meta["record_id"],
            "window_id": meta["window_id"],
            "score": None,
            "predicted_class": predicted_class,
            "ground_truth_class": gt_to_int(meta["label"]),
        })
    return rows


def normalize_dnnpremir(output_path, metadata_by_record):
    rows = []
    current_record_id = None
    with open(output_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_record_id = line[1:]
                continue
            if line.startswith("="):
                continue
            if line.endswith("True") or line.endswith("False"):
                token = line.rsplit(None, 1)[-1]
                predicted_class = 1 if token == "True" else 0
                rows.append({
                    "record_id": current_record_id,
                    "window_id": metadata_by_record[current_record_id]["window_id"],
                    "score": None,
                    "predicted_class": predicted_class,
                    "ground_truth_class": gt_to_int(metadata_by_record[current_record_id]["label"]),
                })

    if len(rows) != len(metadata_by_record):
        raise RuntimeError(f"dnnpremir count mismatch: {len(rows)} predictions vs {len(metadata_by_record)} metadata rows")
    return rows


def normalize_mirdnn(output_path, metadata_by_record, threshold):
    rows = []
    seen = set()
    with open(output_path, newline="") as handle:
        for record_id, score_text in csv.reader(handle):
            if record_id in seen:
                continue
            seen.add(record_id)
            score = float(score_text)
            rows.append({
                "record_id": record_id,
                "window_id": metadata_by_record[record_id]["window_id"],
                "score": score,
                "predicted_class": 1 if score >= threshold else 0,
                "ground_truth_class": gt_to_int(metadata_by_record[record_id]["label"]),
            })

    if len(rows) != len(metadata_by_record):
        raise RuntimeError(f"mirdnn count mismatch: {len(rows)} predictions vs {len(metadata_by_record)} metadata rows")
    return rows


def _infer_mire2e_record_id(window_name, metadata_by_record):
    if window_name in metadata_by_record:
        return window_name
    candidate = window_name.rsplit("-", 2)[0]
    if candidate in metadata_by_record:
        return candidate
    raise KeyError(f"Unable to map miRe2e window '{window_name}' back to a record_id")


def normalize_mire2e(output_path, metadata_by_record, threshold):
    with open(output_path) as handle:
        predictions = json.load(handle)["predictions"]

    best_score = defaultdict(float)
    for pred in predictions:
        record_id = _infer_mire2e_record_id(pred["window"], metadata_by_record)
        score = max(float(pred["score_5_3"]), float(pred["score_3_5"]))
        if score > best_score[record_id]:
            best_score[record_id] = score

    rows = []
    for record_id, meta in metadata_by_record.items():
        score = best_score.get(record_id, 0.0)
        rows.append({
            "record_id": record_id,
            "window_id": meta["window_id"],
            "score": score,
            "predicted_class": 1 if score >= threshold else 0,
            "ground_truth_class": gt_to_int(meta["label"]),
        })
    return rows


def normalize_mustard(output_path, metadata_rows, threshold, positive_column):
    open_fn = gzip.open if str(output_path).endswith(".gz") else open
    preds = []
    with open_fn(output_path, "rt") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            preds.append([float(value) for value in line.split("\t") if value])

    if len(preds) != len(metadata_rows):
        raise RuntimeError(f"mustard count mismatch: {len(preds)} predictions vs {len(metadata_rows)} metadata rows")

    rows = []
    for meta, cols in zip(metadata_rows, preds):
        score = cols[positive_column]
        rows.append({
            "record_id": meta["record_id"],
            "window_id": meta["window_id"],
            "score": score,
            "predicted_class": 1 if score >= threshold else 0,
            "ground_truth_class": gt_to_int(meta["label"]),
        })
    return rows


def normalize_tool_output(tool, output_path, metadata_path, threshold=0.5, mustard_positive_column=0):
    metadata_rows, metadata_by_record = load_metadata(metadata_path)
    output_path = Path(output_path)

    if tool == "deepmir":
        return normalize_deepmir(output_path, metadata_by_record)
    if tool == "deepmirgene":
        return normalize_deepmirgene(output_path, metadata_rows)
    if tool == "dnnpremir":
        return normalize_dnnpremir(output_path, metadata_by_record)
    if tool == "mirdnn":
        return normalize_mirdnn(output_path, metadata_by_record, threshold)
    if tool == "mire2e":
        return normalize_mire2e(output_path, metadata_by_record, threshold)
    if tool == "mustard":
        return normalize_mustard(output_path, metadata_rows, threshold, mustard_positive_column)
    raise RuntimeError(f"Unsupported tool: {tool}")


def resolve_result_path(tool, results_dir, output_name):
    base = Path(results_dir) / tool / output_name
    if tool == "mustard":
        matches = sorted(base.glob("predict/static/results/intermediate_files/*.predictions.txt.gz"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one MuStARD predictions file under {base}, found {len(matches)}")
        return matches[0]

    relative = RESULT_FILES[tool]
    output_path = base / relative
    if not output_path.exists():
        raise FileNotFoundError(f"Missing output for {tool}: {output_path}")
    return output_path
