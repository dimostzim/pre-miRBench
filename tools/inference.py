#!/usr/bin/env python
import argparse
import os
import shutil
import subprocess

try:
    import yaml
except ImportError:
    yaml = None


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
            return yaml.safe_load(f)

    config = {}
    with open(config_path) as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = _parse_scalar(value)
    return config


def _config_repo_path(repo_root, value):
    if os.path.isabs(value):
        return value
    return os.path.join(repo_root, value)


def require_repo_path(repo_root, value, description, expect_dir=False):
    path = _config_repo_path(repo_root, value)
    exists = os.path.isdir(path) if expect_dir else os.path.isfile(path)
    if not exists:
        kind = "directory" if expect_dir else "file"
        raise FileNotFoundError(f"Missing {description} {kind}: {path}")
    return path


def require_mustard_conservation_files(repo_root, cons_dir, chrom_list, input_mode):
    if "conservation" not in [item.strip() for item in input_mode.split(",") if item.strip()]:
        return

    if chrom_list == "all":
        return

    cons_dir_path = _config_repo_path(repo_root, cons_dir)
    for chrom in [item.strip() for item in chrom_list.split(",") if item.strip()]:
        wigfix_path = os.path.join(cons_dir_path, f"{chrom}.wigFix.gz")
        if not os.path.isfile(wigfix_path):
            raise FileNotFoundError(f"Missing MuStARD conservation file: {wigfix_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", required=True, choices=["mustard", "mire2e", "mirdnn", "dnnpremir", "deepmir", "deepmirgene"])
    p.add_argument("--output-name", required=True, help="Subdirectory under results/<tool>/ to store this run")
    p.add_argument("--config", help="Optional explicit config file path; defaults to configs/<tool>_config.yaml")
    args = p.parse_args()

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(tools_dir)
    if args.config:
        config_path = os.path.abspath(args.config)
    else:
        config_path = os.path.join(repo_root, "configs", f"{args.tool}_config.yaml")

    config = load_config(config_path)

    if args.tool == "mustard":
        require_repo_path(repo_root, config["targetIntervals"], "MuStARD targetIntervals")
        require_repo_path(repo_root, config["genome"], "MuStARD genome FASTA")
        require_repo_path(repo_root, config["consDir"], "MuStARD conservation directory", expect_dir=True)
        require_mustard_conservation_files(
            repo_root,
            config["consDir"],
            config["chromList"],
            config["inputMode"],
        )
    else:
        require_repo_path(repo_root, config["input"], f"{args.tool} input")

    if args.tool == "deepmirgene" and config.get("model"):
        require_repo_path(repo_root, config["model"], "deepMiRGene model")

    output_dir = os.path.join(repo_root, "results", args.tool, args.output_name)
    os.makedirs(output_dir, exist_ok=True)
    tool_home_dir = os.path.join(repo_root, "results", args.tool, "_home")
    tool_cache_dir = os.path.join(tool_home_dir, ".cache")
    os.makedirs(tool_home_dir, exist_ok=True)
    os.makedirs(tool_cache_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", f"HOME=/work/results/{args.tool}/_home",
        "-e", f"XDG_CACHE_HOME=/work/results/{args.tool}/_home/.cache",
        "-v", f"{repo_root}:/work",
    ]

    if args.tool == "deepmir":
        deepmir_user_data_dir = os.path.join(repo_root, "results", "deepmir", "_user_data")
        deepmir_runtime_predictor = os.path.join(repo_root, "tools", "deepmir", "runtime_predictor.py")
        deepmir_input_basename = os.path.splitext(os.path.basename(config["input"]))[0]
        deepmir_input_user_data_dir = os.path.join(deepmir_user_data_dir, deepmir_input_basename)
        os.makedirs(deepmir_user_data_dir, exist_ok=True)
        if os.path.isdir(deepmir_input_user_data_dir):
            shutil.rmtree(deepmir_input_user_data_dir)
        cmd.extend(["-v", f"{deepmir_user_data_dir}:/opt/deepmir/deepmir_src/user_data"])
        cmd.extend(["-v", f"{deepmir_runtime_predictor}:/opt/deepmir/deepmir_src/predictor.py:ro"])
    elif args.tool == "deepmirgene":
        deepmirgene_results_dir = os.path.join(repo_root, "results", "deepmirgene", "_scratch_results")
        os.makedirs(deepmirgene_results_dir, exist_ok=True)
        cmd.extend(["-v", f"{deepmirgene_results_dir}:/opt/deepmirgene/deepmirgene_src/inference/results"])
    elif args.tool == "dnnpremir":
        dnnpremir_temp_dir = os.path.join(repo_root, "results", "dnnpremir", "_temp")
        os.makedirs(dnnpremir_temp_dir, exist_ok=True)
        cmd.extend(["-v", f"{dnnpremir_temp_dir}:/opt/dnnpremir/dnnpremir_src/temp"])
    elif args.tool == "mire2e":
        checkpoint_dir = os.path.join(tool_cache_dir, "torch", "hub", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        local_model_dir = os.path.join(repo_root, "tools", "mire2e", "mire2e_src", "models")
        for filename in (
            "structure-hsa.pkl",
            "mfe-hsa.pkl",
            "predictor-hsa.pkl",
            "structure.pkl",
            "mfe.pkl",
            "predictor.pkl",
        ):
            source = os.path.join(local_model_dir, filename)
            target = os.path.join(checkpoint_dir, filename)
            if os.path.exists(source) and not os.path.exists(target):
                shutil.copy2(source, target)

    cmd.append(f"{args.tool}:latest")
    path_prefix = "/work/"
    output_path = f"/work/results/{args.tool}/{args.output_name}"

    if args.tool == "mustard":
        cmd.extend(["--targetIntervals", f"{path_prefix}{config['targetIntervals']}"])
        cmd.extend(["--genome", f"{path_prefix}{config['genome']}"])
        cmd.extend(["--consDir", f"{path_prefix}{config['consDir']}"])
        cmd.extend(["--dir", output_path])
        cmd.extend(["--chromList", config["chromList"]])
        cmd.extend(["--model", config["model"]])
        cmd.extend(["--classNum", str(config["classNum"])])
        cmd.extend(["--modelType", config["modelType"]])
        cmd.extend(["--winSize", str(config["winSize"])])
        cmd.extend(["--step", str(config["step"])])
        cmd.extend(["--staticPredFlag", str(config["staticPredFlag"])])
        cmd.extend(["--inputMode", config["inputMode"]])
        cmd.extend(["--threads", str(config["threads"])])
        cmd.extend(["--modelDirName", config["modelDirName"]])
        cmd.extend(["--intermDir", f"{path_prefix}{config['intermDir']}"])

    elif args.tool == "mire2e":
        cmd.extend(["--input", f"{path_prefix}{config['input']}"])
        cmd.extend(["--output", output_path])
        cmd.extend(["--device", config["device"]])
        cmd.extend(["--pretrained", config["pretrained"]])
        cmd.extend(["--length", str(config["length"])])
        cmd.extend(["--step", str(config["step"])])
        cmd.extend(["--batch_size", str(config["batch_size"])])

    elif args.tool == "mirdnn":
        cmd.extend(["--input", f"{path_prefix}{config['input']}"])
        cmd.extend(["--output", output_path])
        cmd.extend(["--model", config["model"]])
        cmd.extend(["--seq_length", str(config["seq_length"])])
        cmd.extend(["--device", config["device"]])
        cmd.extend(["--batch_size", str(config["batch_size"])])

    elif args.tool == "dnnpremir":
        cmd.extend(["--input", f"{path_prefix}{config['input']}"])
        cmd.extend(["--output", output_path])
        if "seq_length" in config and config["seq_length"] is not None:
            cmd.extend(["--seq_length", str(config["seq_length"])])

    elif args.tool == "deepmir":
        cmd.extend(["--input", f"{path_prefix}{config['input']}"])
        cmd.extend(["--output", output_path])
        cmd.extend(["--model", config["model"]])

    elif args.tool == "deepmirgene":
        cmd.extend(["--input", f"{path_prefix}{config['input']}"])
        cmd.extend(["--output", output_path])
        if config.get("model"):
            cmd.extend(["--model", f"{path_prefix}{config['model']}"])

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
