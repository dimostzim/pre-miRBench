#!/usr/bin/env python3
import argparse
import contextlib
import csv
import gzip
import html
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path


TOOLS = ("deepmir", "deepmirgene", "dnnpremir", "mirdnn", "mire2e", "mustard")
FASTA_TOOLS = {"deepmir", "deepmirgene", "dnnpremir", "mirdnn", "mire2e"}
DEFAULT_SPLITS = (
    "test_known_species_known_family",
    "test_known_species_heldout_family",
    "test_heldout_species_known_family",
    "test_heldout_species_heldout_family",
)
SPLIT_LABELS = {
    "test_known_species_known_family": "Known species / Known family",
    "test_known_species_heldout_family": "Known species / Held-out family",
    "test_heldout_species_known_family": "Held-out species / Known family",
    "test_heldout_species_heldout_family": "Held-out species / Held-out family",
}
SPLIT_COLORS = {
    "test_known_species_known_family": "#2f6fbd",
    "test_known_species_heldout_family": "#6b8f3a",
    "test_heldout_species_known_family": "#d7832f",
    "test_heldout_species_heldout_family": "#9f5f80",
}


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_train_helpers(root):
    module_path = root / "pipeline" / "train.py"
    spec = importlib.util.spec_from_file_location("train_wrapper_helpers", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_csv_list(value, allowed=None):
    if not value or value == "all":
        return list(allowed or [])
    items = [item.strip() for item in value.split(",") if item.strip()]
    if allowed:
        unknown = sorted(set(items) - set(allowed))
        if unknown:
            raise ValueError(f"Unsupported values: {', '.join(unknown)}")
    return items


def expand_path(path):
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def evaluation_output_dir(args):
    root = repo_root()
    return expand_path(args.output_dir) if args.output_dir else root / "results" / "evaluation" / args.run_name


def evaluation_log_path(args, eval_dir):
    if args.log_file:
        return expand_path(args.log_file)
    return eval_dir / "run.log.txt"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def config_host_path(root, value):
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(root / value)


def looks_like_path(value):
    return isinstance(value, str) and (
        "/" in value or "\\" in value or value.endswith((".fa", ".bed", ".h5", ".hdf5", ".pmt", ".pkl"))
    )


def require_config_path(root, config, key, expect_dir=False):
    if not config.get(key):
        raise ValueError(f"Missing required inference config field: {key}")
    path = config_host_path(root, config[key])
    exists = os.path.isdir(path) if expect_dir else os.path.isfile(path)
    if not exists:
        kind = "directory" if expect_dir else "file"
        raise FileNotFoundError(f"Missing {key} {kind}: {path}")
    return path


def optional_path_or_literal(root, train_helpers, config, key, mounts, read_only=True):
    value = config.get(key)
    if not value:
        return None
    if looks_like_path(value):
        path = config_host_path(root, value)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {key} file: {path}")
        return train_helpers.container_path(str(root), path, mounts, read_only=read_only)
    return str(value)


def read_fasta_records(path):
    records = []
    name = None
    parts = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(parts)))
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
    if name is not None:
        records.append((name, "".join(parts)))
    return records


def read_bed_records(path):
    records = []
    with open(path) as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.startswith("#"):
                continue
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) < 6:
                raise ValueError(f"BED rows need at least 6 columns: {path}")
            records.append((fields[3], raw_line.rstrip("\n")))
    return records


def write_fasta(path, records):
    with open(path, "w") as handle:
        for record_id, sequence in records:
            handle.write(f">{record_id}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset:offset + 80] + "\n")


def write_bed(path, records):
    with open(path, "w") as handle:
        for _, line in records:
            handle.write(line + "\n")


def split_input_paths(dataset_dir, tool, split):
    tool_dir = dataset_dir / "tool_inputs" / tool
    suffix = "fa" if tool in FASTA_TOOLS else "bed"
    return (
        tool_dir / f"{split}_positive.{suffix}",
        tool_dir / f"{split}_negative.{suffix}",
    )


def make_labeled_input(dataset_dir, eval_dir, tool, split):
    positive_path, negative_path = split_input_paths(dataset_dir, tool, split)
    if not positive_path.is_file() or not negative_path.is_file():
        raise FileNotFoundError(f"Missing {tool} {split} input files under {positive_path.parent}")

    input_dir = eval_dir / "inputs" / tool
    input_dir.mkdir(parents=True, exist_ok=True)

    if tool in FASTA_TOOLS:
        positive_records = read_fasta_records(positive_path)
        negative_records = read_fasta_records(negative_path)
        input_path = input_dir / f"{split}.fa"
        write_fasta(input_path, positive_records + negative_records)
    else:
        positive_records = read_bed_records(positive_path)
        negative_records = read_bed_records(negative_path)
        input_path = input_dir / f"{split}.bed"
        write_bed(input_path, positive_records + negative_records)

    labels = []
    seen = set()
    for label, records in ((1, positive_records), (0, negative_records)):
        for record_id, _ in records:
            if record_id in seen:
                raise ValueError(f"Duplicate record_id in {tool} {split}: {record_id}")
            seen.add(record_id)
            labels.append({"record_id": record_id, "label": label})
    if not labels:
        raise ValueError(f"No records in {tool} {split} evaluation input")

    labels_path = input_dir / f"{split}.labels.csv"
    write_table(labels_path, labels, ["record_id", "label"])
    return input_path, labels


def write_table(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_table(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value):
    if value in (None, ""):
        return None
    return float(value)


def write_auprc_bar_plot(path, metric_rows, splits=DEFAULT_SPLITS):
    values = {}
    for row in metric_rows:
        auprc = as_float(row.get("auprc"))
        if auprc is None:
            continue
        values[(row["tool"], row["split"])] = auprc

    tools = [tool for tool in TOOLS if any((tool, split) in values for split in splits)]
    if not tools:
        return False

    width = max(900, 170 * len(tools) + 180)
    height = 540
    margin_left = 70
    margin_right = 35
    margin_top = 55
    margin_bottom = 130
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    baseline = margin_top + plot_height
    group_width = plot_width / len(tools)
    split_count = max(1, len(splits))
    bar_width = min(26, group_width * 0.72 / split_count)
    if len(splits) == 1:
        split_offsets = {splits[0]: 0}
    else:
        offset_step = bar_width * 1.16
        midpoint = (len(splits) - 1) / 2.0
        split_offsets = {split: (index - midpoint) * offset_step for index, split in enumerate(splits)}

    def x_for(tool_index, split):
        center = margin_left + group_width * (tool_index + 0.5)
        return center + split_offsets.get(split, 0) - bar_width / 2

    def y_for(value):
        return baseline - max(0.0, min(1.0, value)) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #1f2933; }",
        ".axis { stroke: #2d3748; stroke-width: 1.3; }",
        ".grid { stroke: #d8dee9; stroke-width: 1; }",
        ".tick { font-size: 12px; fill: #52606d; }",
        ".label { font-size: 13px; }",
        ".title { font-size: 18px; font-weight: 700; }",
        ".value { font-size: 11px; fill: #334e68; }",
        "</style>",
        f'<text class="title" x="{width / 2:.1f}" y="28" text-anchor="middle">AUPRC by Tool</text>',
    ]

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_for(tick)
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}"/>')
        lines.append(f'<text class="tick" x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick:.2f}</text>')

    lines.append(f'<line class="axis" x1="{margin_left}" y1="{baseline}" x2="{width - margin_right}" y2="{baseline}"/>')
    lines.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{baseline}"/>')
    lines.append(f'<text class="label" x="{margin_left + plot_width / 2:.1f}" y="{baseline + 58}" text-anchor="middle">Tool</text>')
    lines.append(
        f'<text class="label" x="18" y="{margin_top + plot_height / 2:.1f}" text-anchor="middle" '
        'transform="rotate(-90 18 '
        f'{margin_top + plot_height / 2:.1f})">AUPRC</text>'
    )

    legend_items = [(split, SPLIT_LABELS.get(split, split)) for split in splits]
    legend_widths = [max(90, 28 + 7 * len(label)) for _, label in legend_items]
    legend_x = margin_left + plot_width / 2.0 - sum(legend_widths) / 2.0
    legend_y = baseline + 80
    cursor = legend_x
    for (split, label_text), item_width in zip(legend_items, legend_widths):
        color = SPLIT_COLORS.get(split, "#6b7280")
        label = html.escape(label_text)
        lines.append(f'<rect x="{cursor:.1f}" y="{legend_y:.1f}" width="13" height="13" fill="{color}"/>')
        lines.append(f'<text class="tick" x="{cursor + 18:.1f}" y="{legend_y + 11:.1f}">{label}</text>')
        cursor += item_width

    for tool_index, tool in enumerate(tools):
        center = margin_left + group_width * (tool_index + 0.5)
        lines.append(f'<text class="label" x="{center:.1f}" y="{baseline + 28}" text-anchor="middle">{html.escape(tool)}</text>')
        for split in splits:
            value = values.get((tool, split))
            if value is None:
                continue
            x = x_for(tool_index, split)
            y = y_for(value)
            bar_height = baseline - y
            color = SPLIT_COLORS.get(split, "#6b7280")
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>')
            lines.append(
                f'<text class="value" x="{x + bar_width / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle">'
                f"{value:.3f}</text>"
            )

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return True


def write_auprc_bar_plot_png(path, metric_rows, splits=DEFAULT_SPLITS):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: skipped PNG AUPRC plot because matplotlib is not installed")
        return False

    values = {}
    for row in metric_rows:
        auprc = as_float(row.get("auprc"))
        if auprc is None:
            continue
        values[(row["tool"], row["split"])] = auprc

    tools = [tool for tool in TOOLS if any((tool, split) in values for split in splits)]
    if not tools:
        return False

    fig_width = max(13.0, 2.7 * len(tools) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5.4), dpi=160)
    x_positions = list(range(len(tools)))
    total_group_width = 0.82
    bar_width = total_group_width / max(1, len(splits)) if len(splits) > 1 else 0.55
    midpoint = (len(splits) - 1) / 2.0

    for split_index, split in enumerate(splits):
        offset = (split_index - midpoint) * bar_width
        heights = [values.get((tool, split), 0.0) for tool in tools]
        present = [(tool, split) in values for tool in tools]
        color = SPLIT_COLORS.get(split, "#6b7280")
        bars = ax.bar(
            [position + offset for position in x_positions],
            heights,
            width=bar_width * 0.62,
            color=color,
            label=SPLIT_LABELS.get(split, split),
        )
        for bar, is_present in zip(bars, present):
            if not is_present:
                bar.set_alpha(0.0)
                continue
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.012,
                f"{height * 100:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#000000",
            )

    ax.set_title("AUPRC by Tool", fontsize=18, fontweight="bold", color="#000000")
    ax.set_xlabel("")
    ax.set_ylabel("AUPRC", fontsize=16, color="#000000")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tools, rotation=0, ha="center", fontsize=15, color="#000000")
    ax.tick_params(axis="y", labelsize=13, colors="#000000")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", color="#d8dee9", linewidth=0.8)
    ax.set_axisbelow(True)
    legend = ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2 if len(splits) > 1 else 1,
        fontsize=13,
    )
    for text in legend.get_texts():
        text.set_color("#000000")
    for spine in ax.spines.values():
        spine.set_color("#000000")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def build_inference_args(tool, root, train_helpers, config, input_path, output_dir, mounts, input_arg_override=None):
    input_arg = input_arg_override or train_helpers.container_path(str(root), input_path, mounts)
    output_arg = train_helpers.container_path(str(root), output_dir, mounts, read_only=False)
    args = [f"/opt/{tool}/inference.py"]

    if tool == "mustard":
        genome = require_config_path(root, config, "genome")
        cons_dir = require_config_path(root, config, "consDir", expect_dir=True)
        model = optional_path_or_literal(root, train_helpers, config, "model", mounts)
        interm_dir = output_dir / "intermediate"
        interm_dir.mkdir(parents=True, exist_ok=True)
        interm_arg = train_helpers.container_path(str(root), interm_dir, mounts, read_only=False)
        args.extend(["--targetIntervals", input_arg])
        args.extend(["--genome", train_helpers.container_path(str(root), genome, mounts, read_only=False)])
        args.extend(["--consDir", train_helpers.container_path(str(root), cons_dir, mounts, read_only=False)])
        args.extend(["--dir", output_arg])
        args.extend(["--chromList", str(config.get("chromList", "all"))])
        if model:
            args.extend(["--model", model])
        args.extend(["--classNum", str(config.get("classNum", 2))])
        args.extend(["--modelType", str(config.get("modelType", "CNN"))])
        args.extend(["--winSize", str(config.get("winSize", config.get("maxSize", 200)))])
        args.extend(["--step", str(config.get("step", 5))])
        args.extend(["--staticPredFlag", str(config.get("staticPredFlag", 1))])
        args.extend(["--inputMode", str(config.get("inputMode", "sequence,RNAfold"))])
        args.extend(["--threads", str(config.get("threads", 10))])
        args.extend(["--modelDirName", str(config.get("modelDirName", "results"))])
        args.extend(["--intermDir", interm_arg])
        return args

    args.extend(["--input", input_arg, "--output", output_arg])

    model = optional_path_or_literal(root, train_helpers, config, "model", mounts)
    if model:
        args.extend(["--model", model])

    if tool == "mire2e":
        for key, flag in (
            ("device", "--device"),
            ("pretrained", "--pretrained"),
            ("length", "--length"),
            ("step", "--step"),
            ("batch_size", "--batch_size"),
        ):
            if config.get(key) is not None:
                args.extend([flag, str(config[key])])
        for key, flag in (
            ("structure_model", "--structure_model"),
            ("mfe_model", "--mfe_model"),
            ("predictor_model", "--predictor_model"),
        ):
            value = optional_path_or_literal(root, train_helpers, config, key, mounts)
            if value:
                args.extend([flag, value])
    elif tool == "mirdnn":
        for key, flag in (("seq_length", "--seq_length"), ("device", "--device"), ("batch_size", "--batch_size")):
            if config.get(key) is not None:
                args.extend([flag, str(config[key])])
    elif tool == "dnnpremir" and config.get("seq_length") is not None:
        args.extend(["--seq_length", str(config["seq_length"])])

    return args


def deepmir_runtime_input(input_path, runtime_dir, mounts):
    input_dir = runtime_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    runtime_input = input_dir / input_path.name
    shutil.copy2(input_path, runtime_input)
    mounts[str(input_dir)] = f"{input_dir}:{input_dir}"
    return str(runtime_input)


def add_runtime_mounts(tool, root, eval_dir, split, cmd, reset_runtime=True):
    wrapper = root / "tools" / tool / "inference.py"
    if wrapper.is_file():
        cmd.extend(["-v", f"{wrapper}:/opt/{tool}/inference.py:ro"])

    scratch_dir = eval_dir / "_runtime" / tool / split
    if reset_runtime and scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    if tool == "deepmir":
        user_data = scratch_dir / "user_data"
        user_data.mkdir(parents=True, exist_ok=True)
        runtime_predictor = root / "tools" / "deepmir" / "runtime_predictor.py"
        cmd.extend(["-v", f"{user_data}:/opt/deepmir/deepmir_src/user_data"])
        cmd.extend(["-v", f"{runtime_predictor}:/opt/deepmir/deepmir_src/predictor.py:ro"])
    elif tool == "dnnpremir":
        temp_dir = scratch_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["-v", f"{temp_dir}:/opt/dnnpremir/dnnpremir_src/temp"])


def stream_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_inference(tool, root, train_helpers, config, input_path, output_dir, eval_dir, dry_run=False):
    mounts = {}
    split = input_path.stem
    runtime_dir = eval_dir / "_runtime" / tool / split
    input_arg_override = None
    reset_runtime = True

    if tool == "deepmir":
        if runtime_dir.exists():
            shutil.rmtree(runtime_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        input_arg_override = deepmir_runtime_input(input_path, runtime_dir, mounts)
        reset_runtime = False

    tool_args = build_inference_args(
        tool,
        root,
        train_helpers,
        config,
        input_path,
        output_dir,
        mounts,
        input_arg_override=input_arg_override,
    )

    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-e",
        "XDG_CACHE_HOME=/tmp/.cache",
        "-e",
        "TF_FORCE_GPU_ALLOW_GROWTH=true",
        "-v",
        f"{root}:/work",
    ]
    for mount in mounts.values():
        cmd.extend(["-v", mount])
    add_runtime_mounts(tool, root, eval_dir, split, cmd, reset_runtime=reset_runtime)
    cmd.extend(["--entrypoint", "python", f"{tool}:latest"])
    cmd.extend(tool_args)

    if dry_run:
        print(" ".join(cmd))
        return
    stream_command(cmd)


def parse_deepmir(output_dir):
    result_files = list(output_dir.glob("results.csv")) or list(output_dir.rglob("results.csv"))
    if not result_files:
        raise FileNotFoundError(f"No DeepMir results.csv found under {output_dir}")
    scores = {}
    with open(result_files[0], newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("hairpin"):
                continue
            if row.get("score") not in (None, ""):
                score = float(row["score"])
            else:
                score = 1.0 if row.get("label") == "pre-miRNA" else 0.0
            scores[row["hairpin"]] = score
    return scores


def parse_record_score_table(path, labels):
    with open(path) as handle:
        first = handle.readline()
        handle.seek(0)
        if first.startswith("record_id\t"):
            reader = csv.DictReader(handle, delimiter="\t")
            return {row["record_id"]: float(row["score"]) for row in reader if row.get("record_id")}

        lines = [line.strip() for line in handle if line.strip()]

    expected_ids = [row["record_id"] for row in labels]
    if all(line in {"0", "1"} for line in lines):
        if len(lines) != len(expected_ids):
            raise ValueError(f"{path} has {len(lines)} labels for {len(expected_ids)} input records")
        return {record_id: 1.0 if int(value) == 0 else 0.0 for record_id, value in zip(expected_ids, lines)}

    hard_labels = []
    for line in lines:
        if line in {"True", "False"} or line.endswith(" True") or line.endswith(" False"):
            hard_labels.append(1.0 if line.endswith("True") else 0.0)
    if hard_labels:
        if len(hard_labels) != len(expected_ids):
            raise ValueError(f"{path} has {len(hard_labels)} labels for {len(expected_ids)} input records")
        return dict(zip(expected_ids, hard_labels))

    raise ValueError(f"Could not parse prediction table: {path}")


def parse_mirdnn(output_dir):
    path = output_dir / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No mirDNN predictions.csv found under {output_dir}")
    scores = {}
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) >= 2:
                scores[row[0]] = float(row[1])
    return scores


def parse_mire2e(output_dir, labels):
    path = output_dir / "predictions.json"
    if not path.is_file():
        raise FileNotFoundError(f"No miRe2e predictions.json found under {output_dir}")
    with open(path) as handle:
        payload = json.load(handle)
    predictions = payload.get("predictions", [])
    scores_by_id = defaultdict(list)
    expected_ids = [row["record_id"] for row in labels]

    if predictions and "record_id" in predictions[0]:
        for item in predictions:
            scores_by_id[item["record_id"]].append(max(float(item["score_5_3"]), float(item["score_3_5"])))
    else:
        if len(predictions) != len(expected_ids):
            raise ValueError(f"miRe2e produced {len(predictions)} windows for {len(expected_ids)} input records")
        for record_id, item in zip(expected_ids, predictions):
            scores_by_id[record_id].append(max(float(item["score_5_3"]), float(item["score_3_5"])))

    return {record_id: max(values) for record_id, values in scores_by_id.items()}


def mustard_positive_class_index(config=None):
    if config and config.get("positiveClassIndex") is not None:
        return int(config["positiveClassIndex"])
    return 1


def parse_mustard(output_dir, positive_class_index=1):
    class_name = f"class_{int(positive_class_index)}"
    all_files = list(output_dir.rglob(f"bed_tracks/all.predictions.{class_name}.bed.gz"))
    files = all_files or [
        path
        for path in output_dir.rglob(f"bed_tracks/predictions.*.{class_name}.bed.gz")
        if path.name != f"all.predictions.{class_name}.bed.gz"
    ]
    if not files:
        raise FileNotFoundError(f"No MuStARD {class_name} BED predictions found under {output_dir}")
    print(f"MuStARD positive score source: {class_name} ({len(files)} BED file(s))")
    scores = {}
    for path in files:
        with gzip.open(path, "rt") as handle:
            for raw_line in handle:
                if not raw_line.strip() or raw_line.startswith("track"):
                    continue
                fields = raw_line.rstrip("\n").split("\t")
                if len(fields) >= 5:
                    scores[fields[3]] = float(fields[4])
    return scores


def parse_predictions(tool, output_dir, labels, config=None):
    if tool == "deepmir":
        return parse_deepmir(output_dir)
    if tool in {"deepmirgene", "dnnpremir"}:
        return parse_record_score_table(output_dir / "predictions.txt", labels)
    if tool == "mirdnn":
        return parse_mirdnn(output_dir)
    if tool == "mire2e":
        return parse_mire2e(output_dir, labels)
    if tool == "mustard":
        return parse_mustard(output_dir, mustard_positive_class_index(config))
    raise ValueError(f"Unsupported tool: {tool}")


def species_from_record(record_id):
    if "__" in record_id:
        return record_id.split("__", 1)[0]
    for marker in ("_pos_", "_neg_"):
        if marker in record_id:
            return record_id.split(marker, 1)[0]
    return ""


def prediction_rows(tool, split, labels, scores):
    rows = []
    for index, item in enumerate(labels):
        record_id = item["record_id"]
        score = scores.get(record_id)
        rows.append(
            {
                "tool": tool,
                "split": split,
                "record_id": record_id,
                "species": species_from_record(record_id),
                "label": item["label"],
                "score": "" if score is None else score,
                "predicted_label": "" if score is None else int(score >= 0.5),
                "input_order": index,
            }
        )
    return rows


def average_precision(labels, scores):
    positives = sum(labels)
    if not labels or positives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    hits = 0
    total = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx]:
            hits += 1
            total += hits / rank
    return total / positives


def roc_auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    sorted_items = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    idx = 0
    while idx < len(sorted_items):
        end = idx + 1
        while end < len(sorted_items) and sorted_items[end][1] == sorted_items[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for pos in range(idx, end):
            ranks[sorted_items[pos][0]] = avg_rank
        idx = end

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def confusion_metrics(labels, scores, threshold=0.5):
    predicted = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, yhat in zip(labels, predicted) if y == 1 and yhat == 1)
    tn = sum(1 for y, yhat in zip(labels, predicted) if y == 0 and yhat == 0)
    fp = sum(1 for y, yhat in zip(labels, predicted) if y == 0 and yhat == 1)
    fn = sum(1 for y, yhat in zip(labels, predicted) if y == 1 and yhat == 0)

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    accuracy = (tp + tn) / len(labels) if labels else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else None
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "mcc": mcc,
    }


def best_f1(labels, scores):
    best = {"best_f1": None, "best_threshold": None}
    for threshold in sorted(set(scores), reverse=True):
        f1 = confusion_metrics(labels, scores, threshold)["f1"]
        if f1 is not None and (best["best_f1"] is None or f1 > best["best_f1"]):
            best = {"best_f1": f1, "best_threshold": threshold}
    return best


def metric_row(tool, split, rows, group=None):
    scored = [row for row in rows if row["score"] != ""]
    labels = [int(row["label"]) for row in scored]
    scores = [float(row["score"]) for row in scored]
    metrics = confusion_metrics(labels, scores) if scored else {}
    result = {
        "tool": tool,
        "split": split,
        "group": group or "",
        "records": len(rows),
        "scored_records": len(scored),
        "missing_predictions": len(rows) - len(scored),
        "positives": sum(int(row["label"]) for row in rows),
        "negatives": sum(1 for row in rows if int(row["label"]) == 0),
        "auprc": average_precision(labels, scores) if scored else None,
        "auroc": roc_auc(labels, scores) if scored else None,
    }
    result.update(metrics)
    result.update(best_f1(labels, scores) if scored else {"best_f1": None, "best_threshold": None})
    return result


def grouped_metric_rows(tool, split, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["species"]].append(row)
    return [metric_row(tool, split, group_rows, group=species) for species, group_rows in sorted(grouped.items())]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained tools on held-out splits.")
    parser.add_argument("--dataset-dir", default="data/datasets/mirgenedb_71")
    parser.add_argument("--training-root", default="results/training")
    parser.add_argument("--run-name", default="mirgenedb71_gpu_1to10")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tools", default="all")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--skip-inference", action="store_true", help="Parse existing raw outputs without rerunning Docker.")
    parser.add_argument("--resume", action="store_true", help="Reuse non-empty raw output directories instead of rerunning them.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip tools whose trained inference_config.yaml is missing.")
    parser.add_argument("--dry-run", action="store_true", help="Print Docker commands without running them.")
    parser.add_argument("--plot-only", action="store_true", help="Read <output-dir>/metrics.csv and write AUPRC plots only.")
    parser.add_argument("--log-file", help="Optional log path. Defaults to <output-dir>/run.log.txt.")
    return parser.parse_args()


def load_requested_configs(train_helpers, training_root, run_name, tools, allow_missing=False):
    configs = {}
    missing = []
    for tool in tools:
        config_path = training_root / tool / run_name / "inference_config.yaml"
        if not config_path.is_file():
            missing.append((tool, config_path))
            continue
        configs[tool] = train_helpers.load_config(config_path)

    if missing and not allow_missing:
        details = "\n".join(f"  - {tool}: {path}" for tool, path in missing)
        raise FileNotFoundError(
            "Missing trained inference configs. Train these tools first, use --tools to evaluate a subset, "
            f"or pass --allow-missing to skip them:\n{details}"
        )
    for tool, path in missing:
        print(f"warning: skipping {tool}; missing trained inference config: {path}")
    return configs


def has_raw_output(path):
    return path.is_dir() and any(path.iterdir())


def run_evaluation(args):
    root = repo_root()
    train_helpers = load_train_helpers(root)
    dataset_dir = expand_path(args.dataset_dir)
    training_root = expand_path(args.training_root)
    eval_dir = evaluation_output_dir(args)
    tools = parse_csv_list(args.tools, TOOLS)
    splits = parse_csv_list(args.splits, DEFAULT_SPLITS)
    configs = load_requested_configs(train_helpers, training_root, args.run_name, tools, allow_missing=args.allow_missing)
    tools = [tool for tool in tools if tool in configs]
    if not tools:
        raise ValueError("No tools left to evaluate after filtering missing trained configs")

    all_predictions = []
    metrics = []
    species_metrics = []

    for tool in tools:
        config = configs[tool]

        for split in splits:
            input_path, labels = make_labeled_input(dataset_dir, eval_dir, tool, split)
            output_dir = eval_dir / "raw" / tool / split
            scores = None
            if not args.skip_inference:
                if args.resume and has_raw_output(output_dir) and not args.dry_run:
                    try:
                        scores = parse_predictions(tool, output_dir, labels, config)
                        print(f"### {tool} {split} (resume: using existing raw output)")
                    except (FileNotFoundError, ValueError) as error:
                        print(f"### {tool} {split} (resume: existing raw output incomplete; rerunning)")
                        print(f"resume reason: {error}")
                        shutil.rmtree(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        run_inference(tool, root, train_helpers, config, input_path, output_dir, eval_dir, dry_run=args.dry_run)
                elif args.resume and has_raw_output(output_dir):
                    print(f"### {tool} {split} (resume: using existing raw output)")
                else:
                    print(f"### {tool} {split}")
                    if output_dir.exists():
                        shutil.rmtree(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    run_inference(tool, root, train_helpers, config, input_path, output_dir, eval_dir, dry_run=args.dry_run)
            if args.dry_run:
                continue
            if not has_raw_output(output_dir):
                mode = "--skip-inference" if args.skip_inference else "inference"
                raise FileNotFoundError(f"Missing raw output after {mode}: {output_dir}")

            if scores is None:
                scores = parse_predictions(tool, output_dir, labels, config)
            rows = prediction_rows(tool, split, labels, scores)
            all_predictions.extend(rows)
            metrics.append(metric_row(tool, split, rows))
            species_metrics.extend(grouped_metric_rows(tool, split, rows))
            print(
                f"{tool} {split}: scored {metrics[-1]['scored_records']}/{metrics[-1]['records']} "
                f"AUPRC={metrics[-1]['auprc']}"
            )

    if args.dry_run:
        return

    prediction_fields = ["tool", "split", "record_id", "species", "label", "score", "predicted_label", "input_order"]
    metric_fields = [
        "tool",
        "split",
        "group",
        "records",
        "scored_records",
        "missing_predictions",
        "positives",
        "negatives",
        "auprc",
        "auroc",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "best_f1",
        "best_threshold",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    write_table(eval_dir / "predictions.csv", all_predictions, prediction_fields)
    write_table(eval_dir / "metrics.csv", metrics, metric_fields)
    write_table(eval_dir / "metrics_by_species.csv", species_metrics, metric_fields)
    if write_auprc_bar_plot(eval_dir / "auprc_by_tool.svg", metrics):
        print(f"Wrote AUPRC plot: {eval_dir / 'auprc_by_tool.svg'}")
    if write_auprc_bar_plot_png(eval_dir / "auprc_by_tool.png", metrics):
        print(f"Wrote AUPRC plot: {eval_dir / 'auprc_by_tool.png'}")
    print(f"Wrote evaluation outputs under {eval_dir}")


def main():
    args = parse_args()
    eval_dir = evaluation_output_dir(args)
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_path = evaluation_log_path(args, eval_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", buffering=1) as log_handle:
        stdout = Tee(sys.stdout, log_handle)
        stderr = Tee(sys.stderr, log_handle)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"Writing evaluation log: {log_path}")
            try:
                if args.plot_only:
                    metrics_path = eval_dir / "metrics.csv"
                    if not metrics_path.is_file():
                        raise FileNotFoundError(f"Missing metrics.csv for --plot-only: {metrics_path}")
                    metrics = read_table(metrics_path)
                    splits = parse_csv_list(args.splits, DEFAULT_SPLITS)
                    if write_auprc_bar_plot(eval_dir / "auprc_by_tool.svg", metrics, splits=splits):
                        print(f"Wrote AUPRC plot: {eval_dir / 'auprc_by_tool.svg'}")
                    if write_auprc_bar_plot_png(eval_dir / "auprc_by_tool.png", metrics, splits=splits):
                        print(f"Wrote AUPRC plot: {eval_dir / 'auprc_by_tool.png'}")
                else:
                    run_evaluation(args)
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
