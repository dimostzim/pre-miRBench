#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
    )
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


DEFAULT_DATASETS = [
    "benchmark/output/sample_negatives_output/balanced.csv",
    "benchmark/output/sample_negatives_output/imbalanced.csv",
    "benchmark/output/sample_negatives_output/balanced_collapsed.csv",
    "benchmark/output/sample_negatives_output/imbalanced_collapsed.csv",
]


def parse_label(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"1", "true", "positive", "pos", "yes", "y"}:
        return 1
    if v in {"0", "false", "negative", "neg", "no", "n"}:
        return 0
    return None


def write_fasta(dataset_csv, fasta_path, max_rows=None, seed=42, drop_ambiguous=False):
    labels = {}
    order = []
    lengths = {}
    skipped = 0
    allowed = set("ACGUT")

    if max_rows:
        rng = np.random.default_rng(seed)
        reservoir = []
        with open(dataset_csv, newline="") as f:
            reader = csv.DictReader(f)
            eligible = 0
            for row in reader:
                seq = row["sequence"].strip().replace(" ", "").upper()
                if drop_ambiguous and any(c not in allowed for c in seq):
                    skipped += 1
                    continue
                if eligible < max_rows:
                    reservoir.append((row, seq))
                else:
                    j = int(rng.integers(0, eligible + 1))
                    if j < max_rows:
                        reservoir[j] = (row, seq)
                eligible += 1

        with open(fasta_path, "w") as out:
            for row, seq in reservoir:
                seq_id = row["window_id"]
                out.write(f">{seq_id}\n{seq}\n")
                labels[seq_id] = 1 if row["label"] == "positive" else 0
                order.append(seq_id)
                lengths[seq_id] = len(seq)
        if drop_ambiguous and skipped:
            print(f"[write_fasta] Dropped {skipped} sequences with ambiguous bases from {dataset_csv}")
        return labels, order, lengths, [row for row, _ in reservoir]

    with open(dataset_csv, newline="") as f, open(fasta_path, "w") as out:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row["window_id"]
            seq = row["sequence"].strip().replace(" ", "").upper()
            if drop_ambiguous and any(c not in allowed for c in seq):
                skipped += 1
                continue
            out.write(f">{seq_id}\n{seq}\n")
            labels[seq_id] = 1 if row["label"] == "positive" else 0
            order.append(seq_id)
            lengths[seq_id] = len(seq)

    if drop_ambiguous and skipped:
        print(f"[write_fasta] Dropped {skipped} sequences with ambiguous bases from {dataset_csv}")
    return labels, order, lengths, None


def write_trimmed_fasta(dataset_csv, fasta_path, target_len, label_ids=None, drop_ambiguous=False):
    allowed = set("ACGUT")
    kept = 0
    dropped_ambig = 0
    dropped_short = 0

    with open(dataset_csv, newline="") as f, open(fasta_path, "w") as out:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row["window_id"]
            if label_ids is not None and seq_id not in label_ids:
                continue
            seq = row["sequence"].strip().replace(" ", "").upper()
            if drop_ambiguous and any(c not in allowed for c in seq):
                dropped_ambig += 1
                continue
            if len(seq) < target_len:
                dropped_short += 1
                continue
            if len(seq) > target_len:
                start = (len(seq) - target_len) // 2
                seq = seq[start:start + target_len]
            out.write(f">{seq_id}\n{seq}\n")
            kept += 1

    print(f"[write_trimmed_fasta] Wrote {kept} sequences to {fasta_path}")
    if drop_ambiguous and dropped_ambig:
        print(f"[write_trimmed_fasta] Dropped {dropped_ambig} ambiguous sequences")
    if dropped_short:
        print(f"[write_trimmed_fasta] Dropped {dropped_short} short sequences (<{target_len})")


def write_mirdnn_fold(dataset_csv, fold_path, label_ids=None, drop_ambiguous=False):
    allowed = set("ACGUT")
    kept = 0
    dropped_ambig = 0
    missing = 0

    with open(dataset_csv, newline="") as f, open(fold_path, "w") as out:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row.get("window_id")
            if not seq_id:
                continue
            if label_ids is not None and seq_id not in label_ids:
                continue
            seq = row.get("sequence", "").strip().replace(" ", "").upper()
            if drop_ambiguous and any(c not in allowed for c in seq):
                dropped_ambig += 1
                continue
            struct = row.get("structure", "").strip()
            mfe = row.get("mfe", "")
            if not seq or not struct or mfe == "":
                missing += 1
                continue
            try:
                mfe_val = float(mfe)
                mfe_str = f"{mfe_val:.2f}"
            except ValueError:
                mfe_str = str(mfe)
            out.write(f">{seq_id}\n{seq}\n{struct} ({mfe_str})\n")
            kept += 1

    print(f"[write_mirdnn_fold] Wrote {kept} records to {fold_path}")
    if drop_ambiguous and dropped_ambig:
        print(f"[write_mirdnn_fold] Dropped {dropped_ambig} ambiguous sequences")
    if missing:
        print(f"[write_mirdnn_fold] Skipped {missing} rows missing structure/mfe")


def shard_dataset(dataset_csv, shard_root, shard_size, drop_ambiguous=False, max_rows=None, seed=42, reuse_existing=False):
    os.makedirs(shard_root, exist_ok=True)
    existing = sorted(
        f for f in os.listdir(shard_root)
        if f.startswith("shard_") and f.endswith(".csv")
    )
    if reuse_existing and existing:
        shard_csvs = [os.path.join(shard_root, f) for f in existing]
        labels = {}
        for shard_csv in shard_csvs:
            with open(shard_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    seq_id = row["window_id"]
                    labels[seq_id] = 1 if row["label"] == "positive" else 0
        print(f"[shard_dataset] Reusing {len(shard_csvs)} shards from {shard_root}")
        return shard_csvs, labels

    for name in os.listdir(shard_root):
        if name.startswith("shard_") and (name.endswith(".csv") or os.path.isdir(os.path.join(shard_root, name))):
            path = os.path.join(shard_root, name)
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path, topdown=False):
                    for fname in files:
                        os.remove(os.path.join(root, fname))
                    for dname in dirs:
                        os.rmdir(os.path.join(root, dname))
                os.rmdir(path)
            else:
                os.remove(path)

    allowed = set("ACGUT")
    labels = {}
    shard_csvs = []
    kept = 0
    dropped = 0

    def write_rows(rows, fieldnames):
        nonlocal kept
        shard_idx = 0
        writer = None
        out_f = None
        for row in rows:
            if kept % shard_size == 0:
                if out_f:
                    out_f.close()
                shard_csv = os.path.join(shard_root, f"shard_{shard_idx:05d}.csv")
                shard_csvs.append(shard_csv)
                out_f = open(shard_csv, "w", newline="")
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writeheader()
                shard_idx += 1
            writer.writerow(row)
            seq_id = row["window_id"]
            labels[seq_id] = 1 if row["label"] == "positive" else 0
            kept += 1
        if out_f:
            out_f.close()

    if max_rows:
        rng = np.random.default_rng(seed)
        reservoir = []
        with open(dataset_csv, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                seq = row["sequence"].strip().replace(" ", "").upper()
                if drop_ambiguous and any(c not in allowed for c in seq):
                    dropped += 1
                    continue
                if i < max_rows:
                    reservoir.append(row)
                else:
                    j = int(rng.integers(0, i + 1))
                    if j < max_rows:
                        reservoir[j] = row
        if reservoir:
            write_rows(reservoir, list(reservoir[0].keys()))
    else:
        with open(dataset_csv, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                seq = row["sequence"].strip().replace(" ", "").upper()
                if drop_ambiguous and any(c not in allowed for c in seq):
                    dropped += 1
                    continue
                if kept % shard_size == 0:
                    shard_idx = len(shard_csvs)
                    shard_csv = os.path.join(shard_root, f"shard_{shard_idx:05d}.csv")
                    shard_csvs.append(shard_csv)
                    out_f = open(shard_csv, "w", newline="")
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                seq_id = row["window_id"]
                labels[seq_id] = 1 if row["label"] == "positive" else 0
                kept += 1
                if kept % shard_size == 0:
                    out_f.close()
            if "out_f" in locals() and not out_f.closed:
                out_f.close()

    if drop_ambiguous and dropped:
        print(f"[shard_dataset] Dropped {dropped} ambiguous sequences from {dataset_csv}")
    print(f"[shard_dataset] Wrote {kept} rows into {len(shard_csvs)} shards at {shard_root}")
    return shard_csvs, labels
def parse_mirdnn_predictions(path):
    preds = {}
    with open(path, newline="") as f:
        # mirdnn may output headerless two-column CSV
        first = f.readline()
        if not first:
            return preds
        f.seek(0)

        if "sequence_name" in first and "prediction_score" in first:
            reader = csv.DictReader(f)
            for row in reader:
                seq_id = row.get("sequence_name")
                score = row.get("prediction_score")
                if seq_id is None or score is None:
                    continue
                preds[seq_id] = float(score)
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    preds[row[0]] = float(row[1])
                except ValueError:
                    continue

    return preds


def parse_deepmir_results(path):
    preds = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row.get("hairpin")
            label = parse_label(row.get("label"))
            if seq_id is None or label is None:
                continue
            preds[seq_id] = float(label)
    return preds


def parse_bool_predictions(path):
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            seq_id = parts[0]
            label = parse_label(parts[-1])
            if label is None:
                continue
            preds[seq_id] = float(label)
    return preds


def parse_mire2e_predictions(path, seq_order, seq_lengths, length, step, agg="max", topk=3):
    with open(path) as f:
        data = json.load(f)

    preds_by_seq = defaultdict(list)
    window_items = data.get("predictions", [])

    # build index ranges if window ids are numeric
    counts = []
    for seq_id in seq_order:
        L = seq_lengths[seq_id]
        if L < length:
            counts.append(0)
        else:
            counts.append(((L - length) // step) + 1)

    boundaries = []
    total = 0
    for c in counts:
        boundaries.append((total, total + c))
        total += c

    def map_index(idx):
        # map global window index to sequence id based on boundaries
        if idx < 0 or idx >= total:
            return None
        # linear scan is OK for moderate sizes; use binary search for large
        lo = 0
        hi = len(boundaries) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, end = boundaries[mid]
            if start <= idx < end:
                return seq_order[mid]
            if idx < start:
                hi = mid - 1
            else:
                lo = mid + 1
        return None

    for item in window_items:
        idx = item.get("window")
        score = max(float(item.get("score_5_3", 0.0)), float(item.get("score_3_5", 0.0)))

        seq_id = None
        if isinstance(idx, (list, tuple)) and idx:
            if isinstance(idx[0], str):
                seq_id = idx[0]
        elif isinstance(idx, str):
            seq_id = idx
            parts = idx.rsplit("-", 2)
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                seq_id = parts[0]
        elif isinstance(idx, int):
            seq_id = map_index(idx)

        if seq_id is None:
            continue
        preds_by_seq[seq_id].append(score)

    preds = {}
    for seq_id, scores in preds_by_seq.items():
        if not scores:
            continue
        if agg == "mean":
            preds[seq_id] = float(np.mean(scores))
        elif agg == "topk":
            k = min(len(scores), max(1, int(topk)))
            preds[seq_id] = float(np.mean(sorted(scores, reverse=True)[:k]))
        else:
            preds[seq_id] = float(max(scores))

    return preds


def parse_mustard_predictions(path):
    preds = {}
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            name = parts[3]
            try:
                score = float(parts[4])
            except ValueError:
                continue
            preds[name] = score
    return preds


def parse_mustard_intermediate(pred_path, bed_path, positive_class_idx=1):
    preds = {}

    import gzip

    pred_opener = gzip.open if pred_path.endswith(".gz") else open
    bed_opener = gzip.open if bed_path.endswith(".gz") else open

    with pred_opener(pred_path, "rt") as pred_f, bed_opener(bed_path, "rt") as bed_f:
        for pred_line, bed_line in zip(pred_f, bed_f):
            pred_line = pred_line.strip()
            bed_line = bed_line.strip()
            if not pred_line or not bed_line:
                continue

            pred_parts = pred_line.split()
            bed_parts = bed_line.split("\t")
            if len(bed_parts) < 4:
                continue

            class_idx = positive_class_idx if len(pred_parts) > positive_class_idx else 0
            try:
                score = float(pred_parts[class_idx])
            except (ValueError, IndexError):
                continue

            preds[bed_parts[3]] = score

    return preds


def compute_metrics(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "precision": float(
            (y_pred & y_true).sum() / max(1, y_pred.sum())
        ),
        "recall": float(
            (y_pred & y_true).sum() / max(1, y_true.sum())
        ),
    }
    if metrics["precision"] + metrics["recall"]:
        metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
    else:
        metrics["f1"] = 0.0

    if SKLEARN_AVAILABLE and len(np.unique(y_true)) == 2:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_score))
            metrics["auprc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            metrics["auroc"] = None
            metrics["auprc"] = None
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None

    return metrics


def host_to_container(path, repo_root):
    abs_path = os.path.abspath(path)
    repo_root = os.path.abspath(repo_root)
    if not abs_path.startswith(repo_root + os.sep):
        raise ValueError(f"Path {abs_path} is not under repo root {repo_root}")
    rel = os.path.relpath(abs_path, repo_root)
    return os.path.join("/work", rel)


def run_cmd(cmd, quiet, cwd=None, capture_stderr=False):
    if not quiet and not capture_stderr:
        subprocess.check_call(cmd, cwd=cwd)
        return ""
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        if not capture_stderr:
            if completed.stdout:
                print(completed.stdout)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    if not quiet and completed.stdout:
        print(completed.stdout, end="")
    return completed.stderr or ""


def ensure_device_arg(args_list, device):
    args_list = list(args_list or [])
    if "--device" in args_list:
        return args_list
    return args_list + ["--device", device]


def run_tool(
    tool,
    fasta_path,
    out_dir,
    extra_args,
    runner,
    repo_root,
    mustard_cfg,
    quiet,
    mirdnn_fold=None,
    use_gpu=False,
    capture_stderr=False,
):
    os.makedirs(out_dir, exist_ok=True)

    if runner == "docker":
        docker_prefix = ["docker", "run", "--rm"]
        if use_gpu:
            docker_prefix += ["--gpus", "all"]
        image = f"{tool}:latest"
        container_input = host_to_container(fasta_path, repo_root)
        container_output = host_to_container(out_dir, repo_root)
        if tool == "mustard":
            cmd = [
                *docker_prefix,
                "--entrypoint", "python",
                "-v", f"{repo_root}:/work",
                image,
                "/work/tools/mustard/inference.py",
                "--targetIntervals", host_to_container(mustard_cfg["bed"], repo_root),
                "--genome", host_to_container(mustard_cfg["genome"], repo_root),
                "--consDir", host_to_container(mustard_cfg["cons_dir"], repo_root),
                "--dir", container_output,
                "--chromList", mustard_cfg["chrom_list"],
                "--model", mustard_cfg["model"],
                "--classNum", "2",
                "--modelType", "CNN",
                "--winSize", str(mustard_cfg["win_size"]),
                "--step", str(mustard_cfg["step"]),
                "--staticPredFlag", "1",
                "--inputMode", mustard_cfg["input_mode"],
                "--threads", str(mustard_cfg["threads"]),
                "--modelDirName", mustard_cfg["model_dir"],
            ]
        elif tool == "mire2e":
            cmd = [
                *docker_prefix,
                "--entrypoint", "python",
                "-v", f"{repo_root}:/work",
                image,
                "/work/tools/mire2e/inference.py",
                "--input", container_input,
                "--output", container_output,
            ]
        elif tool == "mirdnn" and mirdnn_fold:
            cmd = [
                *docker_prefix,
                "--entrypoint", "python",
                "-v", f"{repo_root}:/work",
                image,
                "/work/tools/mirdnn/inference.py",
                "--input", container_input,
                "--output", container_output,
                "--input_fold", host_to_container(mirdnn_fold, repo_root),
            ]
        else:
            cmd = [
                *docker_prefix,
                "-v", f"{repo_root}:/work",
                image,
                "--input", container_input,
                "--output", container_output,
            ]
    elif runner == "conda":
        if tool == "mirdnn":
            cmd = ["conda", "run", "-n", "mirdnn", "python", "tools/mirdnn/inference.py",
                   "--input", fasta_path, "--output", out_dir]
            if mirdnn_fold:
                cmd += ["--input_fold", mirdnn_fold]
        elif tool == "dnnpremir":
            cmd = ["conda", "run", "-n", "dnnpremir", "python", "tools/dnnpremir/inference.py",
                   "--input", fasta_path, "--output", out_dir]
        elif tool == "deepmir":
            cmd = ["conda", "run", "-n", "deepmir", "python", "tools/deepmir/inference.py",
                   "--input", fasta_path, "--output", out_dir]
        elif tool == "deepmirgene":
            cmd = ["conda", "run", "-n", "deepmirgene", "python", "tools/deepmirgene/inference.py",
                   "--input", fasta_path, "--output", out_dir]
        elif tool == "mire2e":
            cmd = ["conda", "run", "-n", "mire2e", "python", "tools/mire2e/inference.py",
                   "--input", fasta_path, "--output", out_dir]
        elif tool == "mustard":
            cmd = ["conda", "run", "-n", "mustard", "python", "tools/mustard/inference.py",
                   "--targetIntervals", mustard_cfg["bed"],
                   "--genome", mustard_cfg["genome"],
                   "--consDir", mustard_cfg["cons_dir"],
                   "--dir", out_dir,
                   "--chromList", mustard_cfg["chrom_list"],
                   "--model", mustard_cfg["model"],
                   "--classNum", "2",
                   "--modelType", "CNN",
                   "--winSize", str(mustard_cfg["win_size"]),
                   "--step", str(mustard_cfg["step"]),
                   "--staticPredFlag", "1",
                   "--inputMode", mustard_cfg["input_mode"],
                   "--threads", str(mustard_cfg["threads"]),
                   "--modelDirName", mustard_cfg["model_dir"],
                   ]
        else:
            raise ValueError(f"Unknown tool: {tool}")
    else:
        raise ValueError(f"Unknown runner: {runner}")

    cmd += extra_args or []

    return run_cmd(cmd, quiet, cwd=None, capture_stderr=capture_stderr)


def parse_mire2e_args(tool_args):
    length = 100
    step = 20
    if not tool_args:
        return length, step
    args = tool_args.get("mire2e", [])
    for i, val in enumerate(args):
        if val == "--length" and i + 1 < len(args):
            length = int(args[i + 1])
        if val == "--step" and i + 1 < len(args):
            step = int(args[i + 1])
    return length, step


def resolve_mire2e_length_step(tool_args_list):
    length = 100
    step = 20
    if tool_args_list:
        for i, val in enumerate(tool_args_list):
            if val == "--length" and i + 1 < len(tool_args_list):
                length = int(tool_args_list[i + 1])
            if val == "--step" and i + 1 < len(tool_args_list):
                step = int(tool_args_list[i + 1])
    return length, step


def load_predictions(
    tool,
    out_dir,
    seq_order=None,
    seq_lengths=None,
    tool_args=None,
    mire2e_agg="max",
    mire2e_topk=3,
    mire2e_length=None,
    mire2e_step=None,
    mustard_cfg=None,
):
    if tool == "mirdnn":
        return parse_mirdnn_predictions(os.path.join(out_dir, "predictions.csv"))
    if tool == "dnnpremir":
        return parse_bool_predictions(os.path.join(out_dir, "predictions.txt"))
    if tool == "deepmir":
        return parse_deepmir_results(os.path.join(out_dir, "results.csv"))
    if tool == "deepmirgene":
        return parse_bool_predictions(os.path.join(out_dir, "predictions.txt"))
    if tool == "mire2e":
        if seq_order is None or seq_lengths is None:
            raise ValueError("mire2e requires sequence order/lengths to map window scores.")
        length = mire2e_length if mire2e_length is not None else 100
        step = mire2e_step if mire2e_step is not None else 20
        return parse_mire2e_predictions(
            os.path.join(out_dir, "predictions.json"),
            seq_order,
            seq_lengths,
            length,
            step,
            agg=mire2e_agg,
            topk=mire2e_topk,
        )
    if tool == "mustard":
        if mustard_cfg is None:
            raise ValueError("mustard requires config to locate prediction files.")

        # MuStARD writes under <out_dir>/predict/static/<model_dir>/...
        # Keep legacy fallback for older layouts.
        bed_dirs = [
            os.path.join(out_dir, "predict", "static", mustard_cfg["model_dir"], "bed_tracks"),
            os.path.join(out_dir, "predict", "scan", mustard_cfg["model_dir"], "bed_tracks"),
            os.path.join(out_dir, mustard_cfg["model_dir"], "bed_tracks"),
        ]

        for bed_dir in bed_dirs:
            all_pred = os.path.join(bed_dir, "all.predictions.class_1.bed.gz")
            if os.path.exists(all_pred):
                return parse_mustard_predictions(all_pred)

            preds = {}
            for chrom in mustard_cfg["chrom_list"].split(","):
                chrom = chrom.strip()
                if not chrom:
                    continue
                path = os.path.join(bed_dir, f"predictions.{chrom}.class_1.bed.gz")
                if os.path.exists(path):
                    preds.update(parse_mustard_predictions(path))
            if preds:
                return preds

        # Fallback: map intermediate class scores back to window_id via targets.<chrom>.bed.gz.
        preds = {}
        interm_dirs = [
            os.path.join(out_dir, "predict", "static", mustard_cfg["model_dir"], "intermediate_files"),
            os.path.join(out_dir, "predict", "scan", mustard_cfg["model_dir"], "intermediate_files"),
            os.path.join(out_dir, mustard_cfg["model_dir"], "intermediate_files"),
        ]
        bed_roots = [
            os.path.join(out_dir, "predict", "static"),
            os.path.join(out_dir, "predict", "scan"),
            out_dir,
        ]
        chroms = [c.strip() for c in mustard_cfg["chrom_list"].split(",") if c.strip()]
        for chrom in chroms:
            pred_candidates = [
                os.path.join(interm_dir, f"targets.{chrom}.predictions.txt.gz")
                for interm_dir in interm_dirs
            ] + [
                os.path.join(interm_dir, f"targets.{chrom}.predictions.txt")
                for interm_dir in interm_dirs
            ]
            bed_candidates = [os.path.join(root, f"targets.{chrom}.bed.gz") for root in bed_roots]

            pred_path = next((p for p in pred_candidates if os.path.exists(p)), None)
            bed_path = next((p for p in bed_candidates if os.path.exists(p)), None)
            if pred_path and bed_path:
                preds.update(parse_mustard_intermediate(pred_path, bed_path, positive_class_idx=1))

        return preds
    raise ValueError(f"Unknown tool: {tool}")


OOM_ERROR_MARKERS = (
    "CUDA out of memory",
    "MemoryError",
    "Killed",
    "OOM",
    "out of memory",
    "Cannot allocate memory",
)


def split_fasta(fasta_path, n, out_a, out_b):
    count_a = 0
    count_b = 0
    record_idx = 0
    header = None
    seq_lines = []

    def flush_record(out_f_a, out_f_b):
        nonlocal count_a, count_b, record_idx, header, seq_lines
        if header is None:
            return
        target = out_f_a if record_idx < n else out_f_b
        target.write(header)
        for seq_line in seq_lines:
            target.write(seq_line)
        if record_idx < n:
            count_a += 1
        else:
            count_b += 1
        record_idx += 1
        header = None
        seq_lines = []

    with open(fasta_path) as src, open(out_a, "w") as out_f_a, open(out_b, "w") as out_f_b:
        for line in src:
            if line.startswith(">"):
                flush_record(out_f_a, out_f_b)
                header = line
            else:
                seq_lines.append(line)
        flush_record(out_f_a, out_f_b)

    return count_a, count_b


def split_fasta_by_ids(fasta_path, ids_a, out_a, out_b):
    ids_a = set(ids_a)
    count_a = 0
    count_b = 0
    header = None
    seq_lines = []

    def flush_record(out_f_a, out_f_b):
        nonlocal count_a, count_b, header, seq_lines
        if header is None:
            return
        seq_id = header[1:].strip().split()[0]
        if seq_id in ids_a:
            target = out_f_a
            count_a += 1
        else:
            target = out_f_b
            count_b += 1
        target.write(header)
        for seq_line in seq_lines:
            target.write(seq_line)
        header = None
        seq_lines = []

    with open(fasta_path) as src, open(out_a, "w") as out_f_a, open(out_b, "w") as out_f_b:
        for line in src:
            if line.startswith(">"):
                flush_record(out_f_a, out_f_b)
                header = line
            else:
                seq_lines.append(line)
        flush_record(out_f_a, out_f_b)

    return count_a, count_b


def run_tool_with_fallback(
    tool,
    fasta_path,
    out_dir,
    seq_order,
    seq_lengths,
    extra_args,
    runner,
    repo_root,
    mustard_cfg,
    quiet,
    mirdnn_fold,
    use_gpu,
    depth=0,
    max_depth=6,
    mire2e_agg="max",
    mire2e_topk=3,
    mire2e_length=None,
    mire2e_step=None,
):
    try:
        run_tool(
            tool,
            fasta_path,
            out_dir,
            extra_args,
            runner,
            repo_root,
            mustard_cfg,
            quiet=quiet,
            mirdnn_fold=mirdnn_fold,
            use_gpu=use_gpu,
            capture_stderr=True,
        )
        return load_predictions(
            tool,
            out_dir,
            seq_order=seq_order,
            seq_lengths=seq_lengths,
            mire2e_agg=mire2e_agg,
            mire2e_topk=mire2e_topk,
            mire2e_length=mire2e_length,
            mire2e_step=mire2e_step,
            mustard_cfg=mustard_cfg,
        )
    except subprocess.CalledProcessError as e:
        if tool == "mustard":
            raise
        if depth >= max_depth:
            raise

        stderr_text = ((e.stderr or "") + "\n" + (e.output or "")).strip()
        stderr_lower = stderr_text.lower()
        is_oom = (e.returncode == 137) or any(marker.lower() in stderr_lower for marker in OOM_ERROR_MARKERS)
        if not is_oom:
            raise
        if seq_order is None or len(seq_order) < 2:
            raise

        os.makedirs(out_dir, exist_ok=True)
        n_total = len(seq_order)
        split_n = n_total // 2
        split_a_fasta = os.path.join(out_dir, "bisect_a.fa")
        split_b_fasta = os.path.join(out_dir, "bisect_b.fa")
        count_a, count_b = split_fasta(fasta_path, split_n, split_a_fasta, split_b_fasta)
        if count_a == 0 or count_b == 0:
            raise

        dataset_name = os.path.basename(os.path.dirname(out_dir))
        print(
            f"[bisect] {tool} on {dataset_name}: splitting {count_a + count_b} "
            f"sequences into {count_a} + {count_b} (depth {depth})"
        )

        seq_order_a = seq_order[:count_a]
        seq_order_b = seq_order[count_a:count_a + count_b]
        seq_lengths_a = {seq_id: seq_lengths[seq_id] for seq_id in seq_order_a if seq_id in seq_lengths}
        seq_lengths_b = {seq_id: seq_lengths[seq_id] for seq_id in seq_order_b if seq_id in seq_lengths}

        mirdnn_fold_a = None
        mirdnn_fold_b = None
        if tool == "mirdnn" and mirdnn_fold:
            mirdnn_fold_a = os.path.join(out_dir, "bisect_a.fold")
            mirdnn_fold_b = os.path.join(out_dir, "bisect_b.fold")
            split_fasta_by_ids(mirdnn_fold, seq_order_a, mirdnn_fold_a, mirdnn_fold_b)

        preds_a = run_tool_with_fallback(
            tool,
            split_a_fasta,
            f"{out_dir}_bisect_a",
            seq_order_a,
            seq_lengths_a,
            extra_args,
            runner,
            repo_root,
            mustard_cfg,
            quiet,
            mirdnn_fold_a if mirdnn_fold_a else mirdnn_fold,
            use_gpu,
            depth=depth + 1,
            max_depth=max_depth,
            mire2e_agg=mire2e_agg,
            mire2e_topk=mire2e_topk,
            mire2e_length=mire2e_length,
            mire2e_step=mire2e_step,
        )
        preds_b = run_tool_with_fallback(
            tool,
            split_b_fasta,
            f"{out_dir}_bisect_b",
            seq_order_b,
            seq_lengths_b,
            extra_args,
            runner,
            repo_root,
            mustard_cfg,
            quiet,
            mirdnn_fold_b if mirdnn_fold_b else mirdnn_fold,
            use_gpu,
            depth=depth + 1,
            max_depth=max_depth,
            mire2e_agg=mire2e_agg,
            mire2e_topk=mire2e_topk,
            mire2e_length=mire2e_length,
            mire2e_step=mire2e_step,
        )
        preds_a.update(preds_b)
        return preds_a


def main():
    parser = argparse.ArgumentParser(description="Run tools on datasets and compute metrics.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--tools", nargs="+", default=["mirdnn", "dnnpremir", "deepmir", "deepmirgene", "mire2e", "mustard"])
    parser.add_argument("--work_dir", default="benchmark/output/tool_eval")
    parser.add_argument("--metrics_csv", default="benchmark/output/tool_metrics.csv")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap on rows per dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop_ambiguous", action="store_true",
                        help="Drop sequences containing non-ACGUT bases (e.g. N).")
    parser.add_argument("--skip_run", action="store_true", help="Skip tool execution if outputs already exist.")
    parser.add_argument("--runner", choices=["conda", "docker"], default="conda")
    parser.add_argument("--mire2e_agg", choices=["max", "mean", "topk"], default="max")
    parser.add_argument("--mire2e_topk", type=int, default=3)
    parser.add_argument("--mire2e_max_records", type=int, default=None,
                        help="Limit number of sequences passed to miRe2e (multi-record mode).")
    parser.add_argument("--mustard_genome", default="benchmark/download/data/ce11.fa")
    parser.add_argument("--mustard_cons_dir", default="tools/mustard/data")
    parser.add_argument("--mustard_model", default="MuStARD-mirS")
    parser.add_argument("--mustard_input_mode", default="sequence")
    parser.add_argument("--mustard_threads", type=int, default=4)
    parser.add_argument("--mustard_win_size", type=int, default=None,
                        help="Override MuStARD window size (e.g. 100 for MuStARD-mirS).")
    parser.add_argument("--deepmir_len", type=int, default=None,
                        help="Center-trim DeepMir inputs to this length (e.g. 100).")
    parser.add_argument(
        "--mirdnn_use_fold",
        action="store_true",
        default=True,
        help="Use structure/mfe columns to build mirdnn fold input (skip internal RNAfold). Default: True. Use --no_mirdnn_use_fold to disable.",
    )
    parser.add_argument(
        "--no_mirdnn_use_fold",
        action="store_true",
        help="Force mirdnn to run RNAfold internally instead of using pre-computed structure/mfe columns.",
    )
    parser.add_argument("--cpu_tools", nargs="+", default=[],
                        help="Tools to force on CPU even when GPU is enabled (dnnpremir is always CPU).")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU usage (default: try GPU for all tools).")
    parser.add_argument(
        "--shard_size",
        type=int,
        default=None,
        help="Controls CSV read chunk size for memory management. Tool invocation now always uses the full merged input with automatic bisection fallback on OOM.",
    )
    parser.add_argument("--shard_root", default="benchmark/output/tool_eval_shards",
                        help="Root directory for shard files (default: benchmark/output/tool_eval_shards).")
    parser.add_argument("--reuse_shards", action="store_true",
                        help="Reuse existing shard CSVs if present (skip re-sharding).")
    parser.add_argument("--verbose", action="store_true", help="Show tool stdout/stderr.")
    parser.add_argument("--tool_args", default=None,
                        help="JSON file mapping tool->list of extra args, e.g. {'mirdnn':['--device','cpu']}")
    args = parser.parse_args()
    args.mirdnn_use_fold = args.mirdnn_use_fold and not args.no_mirdnn_use_fold

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    tool_args = {}
    if args.tool_args:
        with open(args.tool_args) as f:
            tool_args = json.load(f)

    mire2e_args_list = tool_args.get("mire2e", [])
    mire2e_length, mire2e_step = resolve_mire2e_length_step(mire2e_args_list)

    rows_out = []

    use_gpu = not args.no_gpu
    # DNNPreMiR is unstable on GPU in this benchmark setup; default to CPU.
    cpu_tools = {"dnnpremir"}
    cpu_tools.update(args.cpu_tools or [])
    total_datasets = len(args.datasets)
    total_tools = len(args.tools)

    for dataset_idx, dataset in enumerate(args.datasets, start=1):
        dataset_name = os.path.splitext(os.path.basename(dataset))[0]
        dataset_dir = os.path.join(args.work_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        if args.shard_size:
            shard_root = os.path.join(args.shard_root, dataset_name, f"shards_{args.shard_size}")
            shard_csvs, labels = shard_dataset(
                dataset,
                shard_root,
                args.shard_size,
                drop_ambiguous=args.drop_ambiguous,
                max_rows=args.max_rows,
                seed=args.seed,
                reuse_existing=args.reuse_shards,
            )
            label_ids = set(labels.keys())

            shard_infos = []
            for shard_idx, shard_csv in enumerate(shard_csvs):
                shard_dir = os.path.join(shard_root, f"shard_{shard_idx:05d}")
                os.makedirs(shard_dir, exist_ok=True)

                shard_fasta = os.path.join(shard_dir, "input.fa")
                shard_labels, seq_order, seq_lengths, _ = write_fasta(
                    shard_csv,
                    shard_fasta,
                    max_rows=None,
                    seed=args.seed,
                    drop_ambiguous=False,
                )
                shard_label_ids = set(shard_labels.keys())

                deepmir_fasta = None
                if "deepmir" in args.tools and args.deepmir_len:
                    deepmir_fasta = os.path.join(shard_dir, f"input_deepmir_{args.deepmir_len}.fa")
                    if not os.path.exists(deepmir_fasta):
                        write_trimmed_fasta(
                            shard_csv,
                            deepmir_fasta,
                            args.deepmir_len,
                            label_ids=shard_label_ids,
                            drop_ambiguous=False,
                        )

                mirdnn_fold = None
                if "mirdnn" in args.tools and args.mirdnn_use_fold:
                    mirdnn_fold = os.path.join(shard_dir, "input_mirdnn.fold")
                    if not os.path.exists(mirdnn_fold):
                        write_mirdnn_fold(
                            shard_csv,
                            mirdnn_fold,
                            label_ids=shard_label_ids,
                            drop_ambiguous=False,
                        )

                mustard_cfg = None
                if "mustard" in args.tools:
                    bed_path = os.path.join(shard_dir, "mustard", "targets.bed")
                    os.makedirs(os.path.dirname(bed_path), exist_ok=True)
                    chrom_set = set()
                    mustard_win = args.mustard_win_size
                    if not os.path.exists(bed_path):
                        with open(shard_csv, newline="") as f, open(bed_path, "w") as bed_out:
                            reader = csv.DictReader(f)
                            for row in reader:
                                start = int(row["start"]) - 1
                                end = int(row["end"])
                                if mustard_win:
                                    center = (start + end) // 2
                                    start = max(0, center - mustard_win // 2)
                                    end = start + mustard_win
                                chrom = row["chrom"]
                                strand = row.get("strand", "+")
                                name = row["window_id"]
                                bed_out.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t{strand}\n")
                                chrom_set.add(chrom)
                    else:
                        with open(bed_path, "r") as bed_in:
                            for line in bed_in:
                                if line.strip():
                                    chrom_set.add(line.split("\t")[0])

                    chrom_list = ",".join(sorted(chrom_set))
                    win_size = mustard_win if mustard_win else (max(seq_lengths.values()) if seq_lengths else 200)
                    mustard_cfg = {
                        "bed": bed_path,
                        "chrom_list": chrom_list,
                        "genome": args.mustard_genome,
                        "cons_dir": args.mustard_cons_dir,
                        "model": args.mustard_model,
                        "input_mode": args.mustard_input_mode,
                        "threads": args.mustard_threads,
                        "win_size": win_size,
                        "step": win_size,
                        "model_dir": "eval",
                    }

                shard_infos.append({
                    "shard_dir": shard_dir,
                    "fasta": shard_fasta,
                    "deepmir_fasta": deepmir_fasta,
                    "mirdnn_fold": mirdnn_fold,
                    "seq_order": seq_order,
                    "seq_lengths": seq_lengths,
                    "mustard_cfg": mustard_cfg,
                })

            fasta_path = os.path.join(dataset_dir, "input.fa")
            seq_order = []
            seq_lengths = {}
            with open(fasta_path, "w") as merged_fa:
                for shard_info in shard_infos:
                    with open(shard_info["fasta"]) as shard_fa:
                        shutil.copyfileobj(shard_fa, merged_fa)
                    seq_order.extend(shard_info["seq_order"])
                    seq_lengths.update(shard_info["seq_lengths"])

            deepmir_fasta = None
            if "deepmir" in args.tools and args.deepmir_len:
                deepmir_fasta = os.path.join(dataset_dir, f"input_deepmir_{args.deepmir_len}.fa")
                with open(deepmir_fasta, "w") as merged_deepmir:
                    for shard_info in shard_infos:
                        if shard_info["deepmir_fasta"] and os.path.exists(shard_info["deepmir_fasta"]):
                            with open(shard_info["deepmir_fasta"]) as shard_deepmir:
                                shutil.copyfileobj(shard_deepmir, merged_deepmir)

            mirdnn_fold = None
            if "mirdnn" in args.tools and args.mirdnn_use_fold:
                mirdnn_fold = os.path.join(dataset_dir, "input_mirdnn.fold")
                with open(mirdnn_fold, "w") as merged_fold:
                    for shard_info in shard_infos:
                        if shard_info["mirdnn_fold"] and os.path.exists(shard_info["mirdnn_fold"]):
                            with open(shard_info["mirdnn_fold"]) as shard_fold:
                                shutil.copyfileobj(shard_fold, merged_fold)

            mustard_cfg = None
            if "mustard" in args.tools:
                bed_path = os.path.join(dataset_dir, "mustard", "targets.bed")
                os.makedirs(os.path.dirname(bed_path), exist_ok=True)
                chrom_set = set()
                with open(bed_path, "w") as merged_bed:
                    for shard_info in shard_infos:
                        shard_cfg = shard_info["mustard_cfg"]
                        if not shard_cfg:
                            continue
                        chrom_set.update(c for c in shard_cfg["chrom_list"].split(",") if c)
                        with open(shard_cfg["bed"]) as shard_bed:
                            shutil.copyfileobj(shard_bed, merged_bed)
                chrom_list = ",".join(sorted(chrom_set))
                win_size = args.mustard_win_size if args.mustard_win_size else (max(seq_lengths.values()) if seq_lengths else 200)
                mustard_cfg = {
                    "bed": bed_path,
                    "chrom_list": chrom_list,
                    "genome": args.mustard_genome,
                    "cons_dir": args.mustard_cons_dir,
                    "model": args.mustard_model,
                    "input_mode": args.mustard_input_mode,
                    "threads": args.mustard_threads,
                    "win_size": win_size,
                    "step": win_size,
                    "model_dir": "eval",
                }

            if set(seq_order) != label_ids:
                raise RuntimeError("Merged FASTA IDs differ from label IDs in shard mode.")

            for tool_idx, tool in enumerate(args.tools, start=1):
                print(f"[{dataset_idx}/{total_datasets}] {dataset_name} -> [{tool_idx}/{total_tools}] {tool}")
                sys.stdout.flush()
                tool_out_dir = os.path.join(dataset_dir, tool)
                status = "ok"
                note = ""

                try:
                    if not args.skip_run or not os.path.exists(tool_out_dir):
                        extra_args = list(tool_args.get(tool, []))
                        tool_use_gpu = use_gpu and tool not in cpu_tools
                        if tool == "mire2e":
                            if args.mire2e_max_records is not None:
                                extra_args += ["--max_records", str(args.mire2e_max_records)]
                            if tool_use_gpu:
                                extra_args = ensure_device_arg(extra_args, "cuda:0")
                        if tool == "mirdnn" and tool_use_gpu:
                            extra_args = ensure_device_arg(extra_args, "cuda:0")
                        tool_fasta = deepmir_fasta if (tool == "deepmir" and deepmir_fasta) else fasta_path
                        preds = run_tool_with_fallback(
                            tool,
                            tool_fasta,
                            tool_out_dir,
                            seq_order=seq_order,
                            seq_lengths=seq_lengths,
                            extra_args=extra_args,
                            runner=args.runner,
                            repo_root=repo_root,
                            mustard_cfg=mustard_cfg,
                            quiet=not args.verbose,
                            mirdnn_fold=mirdnn_fold if tool == "mirdnn" else None,
                            use_gpu=tool_use_gpu,
                            mire2e_agg=args.mire2e_agg,
                            mire2e_topk=args.mire2e_topk,
                            mire2e_length=mire2e_length,
                            mire2e_step=mire2e_step,
                        )
                    else:
                        preds = load_predictions(
                            tool,
                            tool_out_dir,
                            seq_order=seq_order,
                            seq_lengths=seq_lengths,
                            tool_args=tool_args,
                            mire2e_agg=args.mire2e_agg,
                            mire2e_topk=args.mire2e_topk,
                            mire2e_length=mire2e_length,
                            mire2e_step=mire2e_step,
                            mustard_cfg=mustard_cfg,
                        )

                    pred_ids = set(preds.keys())
                    common_ids = list(label_ids & pred_ids)

                    if not common_ids:
                        status = "no_overlap"
                        note = "No matching sequence IDs between labels and predictions."
                        rows_out.append({
                            "dataset": dataset_name,
                            "tool": tool,
                            "status": status,
                            "note": note,
                        })
                        continue

                    y_true = [labels[i] for i in common_ids]
                    y_score = [preds[i] for i in common_ids]

                    metrics = compute_metrics(y_true, y_score)
                    rows_out.append({
                        "dataset": dataset_name,
                        "tool": tool,
                        "status": status,
                        "note": note,
                        "n_total": len(labels),
                        "n_pred": len(preds),
                        "n_eval": len(common_ids),
                        "pos": int(sum(y_true)),
                        "neg": int(len(y_true) - sum(y_true)),
                        **metrics,
                    })
                except NotImplementedError as e:
                    rows_out.append({
                        "dataset": dataset_name,
                        "tool": tool,
                        "status": "unsupported",
                        "note": str(e),
                    })
                except Exception as e:
                    rows_out.append({
                        "dataset": dataset_name,
                        "tool": tool,
                        "status": "error",
                        "note": str(e),
                    })
            continue

        fasta_path = os.path.join(dataset_dir, "input.fa")
        labels, seq_order, seq_lengths, rows_for_bed = write_fasta(
            dataset,
            fasta_path,
            max_rows=args.max_rows,
            seed=args.seed,
            drop_ambiguous=args.drop_ambiguous,
        )
        label_ids = set(labels.keys())
        deepmir_fasta = None
        mirdnn_fold = None
        if "deepmir" in args.tools and args.deepmir_len:
            deepmir_fasta = os.path.join(dataset_dir, f"input_deepmir_{args.deepmir_len}.fa")
            write_trimmed_fasta(
                dataset,
                deepmir_fasta,
                args.deepmir_len,
                label_ids=label_ids,
                drop_ambiguous=args.drop_ambiguous,
            )
        if "mirdnn" in args.tools and args.mirdnn_use_fold:
            mirdnn_fold = os.path.join(dataset_dir, "input_mirdnn.fold")
            write_mirdnn_fold(
                dataset,
                mirdnn_fold,
                label_ids=label_ids,
                drop_ambiguous=args.drop_ambiguous,
            )

        mustard_cfg = None
        if "mustard" in args.tools:
            bed_path = os.path.join(dataset_dir, "mustard", "targets.bed")
            os.makedirs(os.path.dirname(bed_path), exist_ok=True)

            chrom_set = set()
            mustard_win = args.mustard_win_size
            if rows_for_bed is not None:
                with open(bed_path, "w") as bed_out:
                    for row in rows_for_bed:
                        start = int(row["start"]) - 1
                        end = int(row["end"])
                        if mustard_win:
                            center = (start + end) // 2
                            start = max(0, center - mustard_win // 2)
                            end = start + mustard_win
                        chrom = row["chrom"]
                        strand = row.get("strand", "+")
                        name = row["window_id"]
                        bed_out.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t{strand}\n")
                        chrom_set.add(chrom)
            else:
                allowed = set("ACGUT")
                with open(dataset, newline="") as f, open(bed_path, "w") as bed_out:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if args.drop_ambiguous:
                            seq = row["sequence"].strip().replace(" ", "").upper()
                            if any(c not in allowed for c in seq):
                                continue
                        start = int(row["start"]) - 1
                        end = int(row["end"])
                        if mustard_win:
                            center = (start + end) // 2
                            start = max(0, center - mustard_win // 2)
                            end = start + mustard_win
                        chrom = row["chrom"]
                        strand = row.get("strand", "+")
                        name = row["window_id"]
                        bed_out.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t{strand}\n")
                        chrom_set.add(chrom)

            chrom_list = ",".join(sorted(chrom_set))
            win_size = mustard_win if mustard_win else (max(seq_lengths.values()) if seq_lengths else 200)

            mustard_cfg = {
                "bed": bed_path,
                "chrom_list": chrom_list,
                "genome": args.mustard_genome,
                "cons_dir": args.mustard_cons_dir,
                "model": args.mustard_model,
                "input_mode": args.mustard_input_mode,
                "threads": args.mustard_threads,
                "win_size": win_size,
                "step": win_size,
                "model_dir": "eval",
            }

        for tool_idx, tool in enumerate(args.tools, start=1):
            print(f"[{dataset_idx}/{total_datasets}] {dataset_name} -> [{tool_idx}/{total_tools}] {tool}")
            sys.stdout.flush()
            tool_out_dir = os.path.join(dataset_dir, tool)
            status = "ok"
            note = ""

            try:
                if not args.skip_run or not os.path.exists(tool_out_dir):
                    extra_args = list(tool_args.get(tool, []))
                    tool_use_gpu = use_gpu and tool not in cpu_tools
                    if tool == "mire2e":
                        if args.mire2e_max_records is not None:
                            extra_args += ["--max_records", str(args.mire2e_max_records)]
                        if tool_use_gpu:
                            extra_args = ensure_device_arg(extra_args, "cuda:0")
                    if tool == "mirdnn" and tool_use_gpu:
                        extra_args = ensure_device_arg(extra_args, "cuda:0")
                    tool_fasta = deepmir_fasta if (tool == "deepmir" and deepmir_fasta) else fasta_path
                    preds = run_tool_with_fallback(
                        tool,
                        tool_fasta,
                        tool_out_dir,
                        seq_order=seq_order,
                        seq_lengths=seq_lengths,
                        extra_args=extra_args,
                        runner=args.runner,
                        repo_root=repo_root,
                        mustard_cfg=mustard_cfg,
                        quiet=not args.verbose,
                        mirdnn_fold=mirdnn_fold if tool == "mirdnn" else None,
                        use_gpu=tool_use_gpu,
                        mire2e_agg=args.mire2e_agg,
                        mire2e_topk=args.mire2e_topk,
                        mire2e_length=mire2e_length,
                        mire2e_step=mire2e_step,
                    )
                else:
                    preds = load_predictions(
                        tool,
                        tool_out_dir,
                        seq_order=seq_order,
                        seq_lengths=seq_lengths,
                        tool_args=tool_args,
                        mire2e_agg=args.mire2e_agg,
                        mire2e_topk=args.mire2e_topk,
                        mire2e_length=mire2e_length,
                        mire2e_step=mire2e_step,
                        mustard_cfg=mustard_cfg,
                    )
                pred_ids = set(preds.keys())
                common_ids = list(label_ids & pred_ids)

                if not common_ids:
                    status = "no_overlap"
                    note = "No matching sequence IDs between labels and predictions."
                    rows_out.append({
                        "dataset": dataset_name,
                        "tool": tool,
                        "status": status,
                        "note": note,
                    })
                    continue

                y_true = [labels[i] for i in common_ids]
                y_score = [preds[i] for i in common_ids]

                metrics = compute_metrics(y_true, y_score)
                rows_out.append({
                    "dataset": dataset_name,
                    "tool": tool,
                    "status": status,
                    "note": note,
                    "n_total": len(labels),
                    "n_pred": len(preds),
                    "n_eval": len(common_ids),
                    "pos": int(sum(y_true)),
                    "neg": int(len(y_true) - sum(y_true)),
                    **metrics,
                })
            except NotImplementedError as e:
                rows_out.append({
                    "dataset": dataset_name,
                    "tool": tool,
                    "status": "unsupported",
                    "note": str(e),
                })
            except Exception as e:
                rows_out.append({
                    "dataset": dataset_name,
                    "tool": tool,
                    "status": "error",
                    "note": str(e),
                })

    os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
    fieldnames = sorted({k for row in rows_out for k in row.keys()})
    with open(args.metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Metrics written to {args.metrics_csv}")


if __name__ == "__main__":
    main()
