#!/usr/bin/env python
import argparse
import os
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", required=True, choices=["mustard", "mire2e", "mirdnn", "dnnpremir", "deepmir", "deepmirgene"])
    p.add_argument("--output-name", required=True, help="Subdirectory under results/<tool>/ to store this run")
    args = p.parse_args()

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(tools_dir)
    config_path = os.path.join(repo_root, "configs", f"{args.tool}_config.yaml")

    config = load_config(config_path)

    output_dir = os.path.join(repo_root, "results", args.tool, args.output_name)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{repo_root}:/work",
        f"{args.tool}:latest"
    ]
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
