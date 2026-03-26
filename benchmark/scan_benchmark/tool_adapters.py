#!/usr/bin/env python3
import csv
import gzip
import json
import re
from pathlib import Path

TOOLS = ("deepmir", "deepmirgene", "dnnpremir", "mirdnn", "mire2e", "mustard")
WINDOW_TOOLS = {
    "deepmir": 200,
    "deepmirgene": 200,
    "dnnpremir": 180,
    "mirdnn": 160,
}
NATIVE_TOOLS = ("mire2e", "mustard")
RESULT_FILES = {
    "deepmir": "results.csv",
    "deepmirgene": "predictions.txt",
    "dnnpremir": "predictions.txt",
    "mirdnn": "predictions.csv",
    "mire2e": "predictions.json",
}
MIRE2E_WINDOW_RE = re.compile(r"^(?P<chunk_id>.+)-(?P<offset_start>\d+)-(?P<offset_end>\d+)$")


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


def reverse_complement_rna(sequence):
    sequence = normalize_sequence(sequence)
    return sequence.translate(str.maketrans("ACGUN", "UGCAN"))[::-1]


def load_chr_sequence(fasta_path, chrom):
    sequence_parts = []
    active = False
    with open(fasta_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:].split()[0]
                if active:
                    break
                active = header == chrom
                continue
            if active:
                sequence_parts.append(line.upper())

    if not sequence_parts:
        raise FileNotFoundError(f"Unable to find chromosome '{chrom}' in {fasta_path}")
    return "".join(sequence_parts)


def load_truth_rows(path, chrom=None):
    rows = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            row_chrom = parts[0]
            if chrom and row_chrom != chrom:
                continue
            start = int(parts[1]) + 1
            end = int(parts[2])
            name = parts[3] if len(parts) >= 4 else f"{row_chrom}:{start}-{end}"
            strand = parts[5] if len(parts) >= 6 else "."
            rows.append({
                "chrom": row_chrom,
                "start": start,
                "end": end,
                "name": name,
                "strand": strand,
            })
    rows.sort(key=lambda row: (row["chrom"], row["start"], row["end"], row["name"]))
    return rows


def write_truth_bed(path, truth_rows):
    with open(path, "w") as handle:
        for row in truth_rows:
            handle.write(
                f"{row['chrom']}\t{row['start'] - 1}\t{row['end']}\t{row['name']}\t0\t{row['strand']}\n"
            )


def write_csv_rows(path, fieldnames, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_yaml_config(path, mapping):
    def yaml_scalar(value):
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(str(value))

    with open(path, "w") as handle:
        for key, value in mapping.items():
            handle.write(f"{key}: {yaml_scalar(value)}\n")


def load_chunk_manifest(path):
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "chunk_id": row["chunk_id"],
                "chrom": row["chrom"],
                "core_start": int(row["core_start"]),
                "core_end": int(row["core_end"]),
                "native_start": int(row["native_start"]),
                "native_end": int(row["native_end"]),
            })
    by_id = {row["chunk_id"]: row for row in rows}
    return rows, by_id


def load_window_metadata(path):
    rows = []
    by_record = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {
                "record_id": row["record_id"],
                "chunk_id": row["chunk_id"],
                "chrom": row["chrom"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "strand": row["strand"],
            }
            rows.append(parsed)
            by_record[parsed["record_id"]] = parsed
    return rows, by_record


def iter_global_window_positions(seq_len, window_len, stride, region_start, region_end):
    last_possible_start = seq_len - window_len + 1
    if last_possible_start < region_start:
        return

    remainder = (region_start - 1) % stride
    start = region_start if remainder == 0 else region_start + (stride - remainder)
    while start <= region_end and start <= last_possible_start:
        yield start, start + window_len - 1
        start += stride


def resolve_scan_result_path(tool, results_dir, output_name):
    base = Path(results_dir) / tool / output_name
    if tool == "mustard":
        matches = sorted(base.glob("predict/scan/results/bedGraph_tracks/*.class_0.bed.gz"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one MuStARD scan BED under {base}, found {len(matches)}"
            )
        return matches[0]

    output_path = base / RESULT_FILES[tool]
    if not output_path.exists():
        raise FileNotFoundError(f"Missing output for {tool}: {output_path}")
    return output_path


def _score_value(row):
    if row["score"] is None:
        return float(row["predicted_class"])
    return float(row["score"])


def _scan_row(meta, predicted_class, score=None):
    return {
        "chrom": meta["chrom"],
        "start": int(meta["start"]),
        "end": int(meta["end"]),
        "strand": meta["strand"],
        "score": score,
        "predicted_class": int(predicted_class),
    }


def normalize_deepmir_scan(output_path, metadata_by_record):
    rows = []
    with open(output_path, newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = row["hairpin"]
            meta = metadata_by_record[record_id]
            rows.append(_scan_row(meta, 1 if row["label"] == "pre-miRNA" else 0))
    return rows


def normalize_deepmirgene_scan(output_path, metadata_rows):
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
        rows.append(_scan_row(meta, 1 if pred == "0" else 0))
    return rows


def normalize_dnnpremir_scan(output_path, metadata_by_record):
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
                rows.append(_scan_row(metadata_by_record[current_record_id], 1 if token == "True" else 0))

    if len(rows) != len(metadata_by_record):
        raise RuntimeError(f"dnnpremir count mismatch: {len(rows)} predictions vs {len(metadata_by_record)} metadata rows")
    return rows


def normalize_mirdnn_scan(output_path, metadata_by_record, threshold):
    rows = []
    seen = set()
    with open(output_path, newline="") as handle:
        for record_id, score_text in csv.reader(handle):
            if record_id in seen:
                continue
            seen.add(record_id)
            score = float(score_text)
            rows.append(_scan_row(metadata_by_record[record_id], 1 if score >= threshold else 0, score))

    if len(rows) != len(metadata_by_record):
        raise RuntimeError(f"mirdnn count mismatch: {len(rows)} predictions vs {len(metadata_by_record)} metadata rows")
    return rows


def normalize_mire2e_scan(output_path, chunk_row, threshold):
    with open(output_path) as handle:
        predictions = json.load(handle)["predictions"]

    rows = []
    for pred in predictions:
        match = MIRE2E_WINDOW_RE.match(pred["window"])
        if not match:
            raise ValueError(f"Unexpected miRe2e window name: {pred['window']}")
        if match.group("chunk_id") != chunk_row["chunk_id"]:
            raise ValueError(
                f"miRe2e window chunk mismatch: expected {chunk_row['chunk_id']}, got {match.group('chunk_id')}"
            )

        offset_start = int(match.group("offset_start"))
        offset_end = int(match.group("offset_end"))
        start = chunk_row["native_start"] + offset_start
        end = chunk_row["native_start"] + offset_end - 1

        plus_score = float(pred["score_5_3"])
        minus_score = float(pred["score_3_5"])
        base_meta = {"chrom": chunk_row["chrom"], "start": start, "end": end}
        rows.append(_scan_row({**base_meta, "strand": "+"}, 1 if plus_score >= threshold else 0, plus_score))
        rows.append(_scan_row({**base_meta, "strand": "-"}, 1 if minus_score >= threshold else 0, minus_score))
    return rows


def normalize_mustard_scan(output_path, threshold):
    rows = []
    with gzip.open(output_path, "rt") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            chrom, start_text, end_text, _name, score_text, strand = line.split("\t")[:6]
            score = float(score_text)
            rows.append({
                "chrom": chrom,
                "start": int(start_text) + 1,
                "end": int(end_text),
                "strand": strand,
                "score": score,
                "predicted_class": 1 if score >= threshold else 0,
            })
    return rows


def deduplicate_window_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["chrom"], row["start"], row["end"], row["strand"])
        current = grouped.get(key)
        if current is None:
            grouped[key] = dict(row)
            continue

        current["predicted_class"] = max(current["predicted_class"], row["predicted_class"])
        current_score = current["score"]
        row_score = row["score"]
        if current_score is None:
            current["score"] = row_score
        elif row_score is not None:
            current["score"] = max(float(current_score), float(row_score))

    return [grouped[key] for key in sorted(grouped)]


def merge_positive_windows(rows, merge_gap=0):
    positives = [row for row in rows if row["predicted_class"] == 1]
    positives.sort(key=lambda row: (row["chrom"], row["strand"], row["start"], row["end"]))

    loci = []
    current = None
    for row in positives:
        row_score = _score_value(row)
        if current is None:
            current = {
                "chrom": row["chrom"],
                "start": row["start"],
                "end": row["end"],
                "strand": row["strand"],
                "max_score": row_score,
                "window_count": 1,
            }
            continue

        can_merge = (
            row["chrom"] == current["chrom"]
            and row["strand"] == current["strand"]
            and row["start"] <= current["end"] + merge_gap + 1
        )
        if can_merge:
            current["end"] = max(current["end"], row["end"])
            current["max_score"] = max(current["max_score"], row_score)
            current["window_count"] += 1
            continue

        loci.append(current)
        current = {
            "chrom": row["chrom"],
            "start": row["start"],
            "end": row["end"],
            "strand": row["strand"],
            "max_score": row_score,
            "window_count": 1,
        }

    if current is not None:
        loci.append(current)
    return loci


def _overlap_length(a_start, a_end, b_start, b_end):
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end < start:
        return 0
    return end - start + 1


def evaluate_loci(predicted_loci, truth_rows, chrom_size_bp, overlap_fraction=0.5):
    matched_truth = set()
    matched_predicted = []

    ordered_loci = sorted(
        predicted_loci,
        key=lambda row: (-row["max_score"], row["chrom"], row["start"], row["end"], row["strand"]),
    )

    for locus in ordered_loci:
        best_truth_index = None
        best_overlap = 0
        for idx, truth in enumerate(truth_rows):
            if idx in matched_truth:
                continue
            if locus["chrom"] != truth["chrom"]:
                continue
            if truth["strand"] not in (".", "?") and locus["strand"] != truth["strand"]:
                continue
            overlap = _overlap_length(locus["start"], locus["end"], truth["start"], truth["end"])
            if overlap == 0:
                continue
            truth_length = truth["end"] - truth["start"] + 1
            if overlap / truth_length < overlap_fraction:
                continue
            if overlap > best_overlap:
                best_overlap = overlap
                best_truth_index = idx

        if best_truth_index is not None:
            matched_truth.add(best_truth_index)
            matched_predicted.append(locus)

    predicted_count = len(predicted_loci)
    true_positive_count = len(matched_truth)
    false_positive_count = predicted_count - true_positive_count
    truth_count = len(truth_rows)
    precision = true_positive_count / predicted_count if predicted_count else 0.0
    recall = true_positive_count / truth_count if truth_count else 0.0
    fp_per_mb = false_positive_count / (chrom_size_bp / 1_000_000.0) if chrom_size_bp else 0.0

    return {
        "truth_loci": truth_count,
        "predicted_loci": predicted_count,
        "matched_truth_loci": true_positive_count,
        "false_positive_loci": false_positive_count,
        "precision_locus": precision,
        "locus_recall": recall,
        "fp_per_mb": fp_per_mb,
    }

