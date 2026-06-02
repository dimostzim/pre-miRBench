#!/usr/bin/env python
import argparse
import os
import subprocess

try:
    import yaml
except ImportError:
    yaml = None


TOOLS = ["mustard", "mire2e", "mirdnn", "dnnpremir", "deepmir", "deepmirgene"]


def _parse_scalar(value):
    value = value.strip()
    if not value:
        return ""
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(config_path):
    if yaml is not None:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    config = {}
    with open(config_path) as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = _parse_scalar(value)
    return config


def write_config(config_path, config):
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    if yaml is not None:
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return

    with open(config_path, "w") as f:
        for key, value in config.items():
            if value is None:
                rendered = "null"
            else:
                rendered = str(value)
            f.write(f"{key}: {rendered}\n")


def host_path(repo_root, value):
    if value is None:
        return None
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(repo_root, value))


def require_file(repo_root, config, key):
    value = config.get(key)
    if not value:
        raise ValueError(f"Missing required config field: {key}")
    path = host_path(repo_root, value)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file for {key}: {path}")
    return path


def require_dir(repo_root, config, key):
    value = config.get(key)
    if not value:
        raise ValueError(f"Missing required config field: {key}")
    path = host_path(repo_root, value)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing directory for {key}: {path}")
    return path


def optional_file(repo_root, config, key):
    value = config.get(key)
    if not value:
        return None
    path = host_path(repo_root, value)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file for {key}: {path}")
    return path


def add_mount(mounts, path, read_only=True):
    path = os.path.abspath(path)
    if path in mounts:
        return
    suffix = ":ro" if read_only else ""
    mounts[path] = f"{path}:{path}{suffix}"


def container_path(repo_root, path, mounts):
    path = os.path.abspath(path)
    try:
        rel = os.path.relpath(path, repo_root)
    except ValueError:
        rel = None

    if rel is not None and not rel.startswith("..") and rel != os.pardir:
        return "/work/" + rel.replace(os.sep, "/")

    add_mount(mounts, path)
    return path


def rel_repo_path(repo_root, path):
    rel = os.path.relpath(os.path.abspath(path), repo_root)
    return rel.replace(os.sep, "/")


def add_optional_arg(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def add_common_training_args(cmd, config):
    for key, flag in (
        ("device", "--device"),
        ("batch_size", "--batch_size"),
        ("epochs", "--epochs"),
        ("seed", "--seed"),
        ("validation_split", "--validation_split"),
    ):
        add_optional_arg(cmd, flag, config.get(key))


def build_tool_args(tool, repo_root, config, output_dir, mounts):
    output_arg = "/work/" + rel_repo_path(repo_root, output_dir)
    cmd = ["/opt/{}/train.py".format(tool), "--output", output_arg]

    if tool in {"mire2e", "mirdnn", "deepmir", "deepmirgene"}:
        positive = require_file(repo_root, config, "positive_fasta")
        negative = require_file(repo_root, config, "negative_fasta")
        cmd.extend(["--positive_fasta", container_path(repo_root, positive, mounts)])
        cmd.extend(["--negative_fasta", container_path(repo_root, negative, mounts)])
        for key, flag in (
            ("validation_positive_fasta", "--validation_positive_fasta"),
            ("validation_negative_fasta", "--validation_negative_fasta"),
        ):
            path = optional_file(repo_root, config, key)
            if path:
                cmd.extend([flag, container_path(repo_root, path, mounts)])

        if tool == "deepmir":
            for key, flag in (
                ("pretrain_positive_fasta", "--pretrain_positive_fasta"),
                ("pretrain_negative_fasta", "--pretrain_negative_fasta"),
            ):
                path = optional_file(repo_root, config, key)
                if path:
                    cmd.extend([flag, container_path(repo_root, path, mounts)])

    if tool == "mire2e":
        for key, flag in (
            ("pretrained", "--pretrained"),
            ("train_structure", "--train_structure"),
            ("train_mfe", "--train_mfe"),
            ("length", "--length"),
            ("structure_model", "--structure_model"),
            ("mfe_model", "--mfe_model"),
            ("predictor_model", "--predictor_model"),
            ("structure_training_fasta", "--structure_training_fasta"),
            ("mfe_training_fasta", "--mfe_training_fasta"),
        ):
            value = config.get(key)
            if (key.endswith("_model") or key.endswith("_fasta")) and value:
                value = container_path(repo_root, require_file(repo_root, config, key), mounts)
            add_optional_arg(cmd, flag, value)
        for key, flag in (
            ("structure_batch_size", "--structure_batch_size"),
            ("mfe_batch_size", "--mfe_batch_size"),
            ("structure_epochs", "--structure_epochs"),
            ("mfe_epochs", "--mfe_epochs"),
        ):
            add_optional_arg(cmd, flag, config.get(key))
        add_common_training_args(cmd, config)

    elif tool == "mirdnn":
        for key, flag in (
            ("seq_length", "--seq_length"),
            ("early_stop", "--early_stop"),
            ("valid_prop", "--valid_prop"),
            ("upsample", "--upsample"),
            ("focal_loss", "--focal_loss"),
        ):
            add_optional_arg(cmd, flag, config.get(key))
        add_common_training_args(cmd, config)

    elif tool == "dnnpremir":
        if config.get("positive_csv") and config.get("negative_csv"):
            positive = require_file(repo_root, config, "positive_csv")
            negative = require_file(repo_root, config, "negative_csv")
            cmd.extend(["--positive_csv", container_path(repo_root, positive, mounts)])
            cmd.extend(["--negative_csv", container_path(repo_root, negative, mounts)])
            for key, flag in (
                ("validation_positive_csv", "--validation_positive_csv"),
                ("validation_negative_csv", "--validation_negative_csv"),
            ):
                path = optional_file(repo_root, config, key)
                if path:
                    cmd.extend([flag, container_path(repo_root, path, mounts)])
        else:
            positive = require_file(repo_root, config, "positive_fasta")
            negative = require_file(repo_root, config, "negative_fasta")
            cmd.extend(["--positive_fasta", container_path(repo_root, positive, mounts)])
            cmd.extend(["--negative_fasta", container_path(repo_root, negative, mounts)])
            for key, flag in (
                ("validation_positive_fasta", "--validation_positive_fasta"),
                ("validation_negative_fasta", "--validation_negative_fasta"),
            ):
                path = optional_file(repo_root, config, key)
                if path:
                    cmd.extend([flag, container_path(repo_root, path, mounts)])
        for key, flag in (("architecture", "--architecture"),):
            add_optional_arg(cmd, flag, config.get(key))
        add_common_training_args(cmd, config)

    elif tool == "deepmir":
        for key, flag in (
            ("architecture", "--architecture"),
            ("training_mode", "--training_mode"),
            ("modules", "--modules"),
            ("dense_units", "--dense_units"),
            ("filters", "--filters"),
            ("pretrain_epochs", "--pretrain_epochs"),
        ):
            add_optional_arg(cmd, flag, config.get(key))
        add_common_training_args(cmd, config)

    elif tool == "deepmirgene":
        add_common_training_args(cmd, config)

    elif tool == "mustard":
        for key, flag, expect_dir in (
            ("positiveIntervals", "--positiveIntervals", False),
            ("negativeIntervals", "--negativeIntervals", False),
            ("testPositiveIntervals", "--testPositiveIntervals", False),
            ("testNegativeIntervals", "--testNegativeIntervals", False),
            ("validationPositiveIntervals", "--validationPositiveIntervals", False),
            ("validationNegativeIntervals", "--validationNegativeIntervals", False),
            ("genome", "--genome", False),
            ("consDir", "--consDir", True),
        ):
            if key not in config or config.get(key) is None:
                continue
            path = require_dir(repo_root, config, key) if expect_dir else require_file(repo_root, config, key)
            cmd.extend([flag, container_path(repo_root, path, mounts)])
        for key, flag in (
            ("device", "--device"),
            ("classList", "--classList"),
            ("maxSize", "--maxSize"),
            ("extFlag", "--extFlag"),
            ("reinfNum", "--reinfNum"),
            ("shufClassFlag", "--shufClassFlag"),
            ("inputMode", "--inputMode"),
            ("modelType", "--modelType"),
            ("threads", "--threads"),
            ("excludedTestChroms", "--exclTest"),
            ("excludedValidChroms", "--exclValid"),
        ):
            add_optional_arg(cmd, flag, config.get(key))

    return cmd


def generated_inference_config(tool, repo_root, config, output_dir):
    artifact = rel_repo_path(repo_root, output_dir)
    inference_input = config.get("inference_input")
    fasta_input = inference_input or "data/smoke/tool_smoke_test.fa"

    if tool == "mire2e":
        return {
            "input": fasta_input,
            "device": config.get("device", "cpu"),
            "pretrained": "no",
            "structure_model": f"{artifact}/structure.pkl",
            "mfe_model": f"{artifact}/mfe.pkl",
            "predictor_model": f"{artifact}/predictor.pkl",
            "length": config.get("length", 100),
            "step": config.get("step", 20),
            "batch_size": config.get("inference_batch_size", config.get("batch_size", 4096)),
        }
    if tool == "mirdnn":
        return {
            "input": fasta_input,
            "model": f"{artifact}/model.pmt",
            "device": config.get("device", "cpu"),
            "seq_length": config.get("seq_length", 160),
            "batch_size": config.get("inference_batch_size", config.get("batch_size", 1024)),
        }
    if tool == "dnnpremir":
        return {
            "input": inference_input or config.get("inference_fasta", "data/smoke/tool_smoke_test.fa"),
            "model": f"{artifact}/CNN_model.h5",
            "seq_length": 180,
        }
    if tool == "deepmir":
        return {
            "input": fasta_input,
            "model": f"{artifact}/model.h5",
        }
    if tool == "deepmirgene":
        return {
            "input": fasta_input,
            "model": f"{artifact}/new_test.hdf5",
        }
    if tool == "mustard":
        return {
            "targetIntervals": inference_input or config["positiveIntervals"],
            "genome": config["genome"],
            "consDir": config["consDir"],
            "chromList": config.get("chromList", "all"),
            "model": f"{artifact}/CNNonRaw.hdf5",
            "classNum": config.get("classNum", 2),
            "modelType": config.get("modelType", "CNN"),
            "winSize": config.get("winSize", config.get("maxSize", 200)),
            "step": config.get("step", 5),
            "staticPredFlag": config.get("staticPredFlag", 1),
            "inputMode": config.get("inputMode", "sequence,RNAfold,conservation"),
            "threads": config.get("threads", 10),
            "modelDirName": config.get("modelDirName", "results"),
            "intermDir": config.get("intermDir", "results/mustard_intermediate"),
        }
    raise ValueError(f"Unsupported tool: {tool}")


def main():
    parser = argparse.ArgumentParser(description="Retrain a pre-miRNA prediction tool in Docker.")
    parser.add_argument("--tool", required=True, choices=TOOLS)
    parser.add_argument("--run-name", required=True, help="Subdirectory under results/training/<tool>/")
    parser.add_argument("--config", help="Defaults to configs/train/<tool>_train.yaml")
    args = parser.parse_args()

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(tools_dir)
    config_path = os.path.abspath(args.config) if args.config else os.path.join(
        repo_root, "configs", "train", f"{args.tool}_train.yaml"
    )
    config = load_config(config_path)

    output_dir = os.path.join(repo_root, "results", "training", args.tool, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    mounts = {}
    tool_home_dir = os.path.join(repo_root, "results", "training", args.tool, "_home")
    tool_cache_dir = os.path.join(tool_home_dir, ".cache")
    os.makedirs(tool_cache_dir, exist_ok=True)

    tool_args = build_tool_args(args.tool, repo_root, config, output_dir, mounts)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", f"HOME=/work/results/training/{args.tool}/_home",
        "-e", f"XDG_CACHE_HOME=/work/results/training/{args.tool}/_home/.cache",
        "-v", f"{repo_root}:/work",
    ]
    for mount in mounts.values():
        cmd.extend(["-v", mount])
    cmd.extend(["--entrypoint", "python", f"{args.tool}:latest"])
    cmd.extend(tool_args)

    subprocess.check_call(cmd)

    inference_config = generated_inference_config(args.tool, repo_root, config, output_dir)
    write_config(os.path.join(output_dir, "inference_config.yaml"), inference_config)
    print(f"Wrote generated inference config: {os.path.join(output_dir, 'inference_config.yaml')}")


if __name__ == "__main__":
    main()
