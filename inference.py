#!/usr/bin/env python
import argparse
import os
import subprocess
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", required=True, choices=["mustard", "mire2e", "mirdnn"])
    p.add_argument("--docker", action="store_true")
    p.add_argument("--output-name", required=True, help="Subdirectory under results/<tool>/ to store this run")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "configs", f"{args.tool}_config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = os.path.join(base_dir, "results", args.tool, args.output_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.docker:
        cmd = [
            "docker", "run", "--rm", "--platform", "linux/amd64",
            "-v", f"{base_dir}:/work",
            f"{args.tool}:latest"
        ]
        path_prefix = "/work/"
        output_path = f"/work/results/{args.tool}/{args.output_name}"
    else:
        conda_base = subprocess.check_output(["conda", "info", "--base"], text=True).strip()
        conda_python = os.path.join(conda_base, "envs", args.tool, "bin", "python")
        tool_script = os.path.join(base_dir, "tools", args.tool, "inference.py")
        cmd = [conda_python, tool_script]
        path_prefix = ""
        output_path = output_dir

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
        if config["intermDir"] != "same":
            cmd.extend(["--intermDir", f"{path_prefix}{config['intermDir']}"])

    elif args.tool == "mire2e":
        cmd.extend(["--input", f"{path_prefix}{config['input']}"])
        cmd.extend(["--output", output_path])
        cmd.extend(["--device", config["device"]])
        cmd.extend(["--pretrained", config["pretrained"]])
        cmd.extend(["--length", str(config["length"])])
        cmd.extend(["--step", str(config["step"])])
        cmd.extend(["--batch_size", str(config["batch_size"])])
        if config["verbose"]:
            cmd.append("--verbose")

    elif args.tool == "mirdnn":
        cmd.extend(["--input", f"{path_prefix}{config['input']}"])
        cmd.extend(["--output", output_path])
        cmd.extend(["--model", config["model"]])
        cmd.extend(["--seq_length", str(config["seq_length"])])
        cmd.extend(["--device", config["device"]])
        cmd.extend(["--batch_size", str(config["batch_size"])])

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
