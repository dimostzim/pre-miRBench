#!/usr/bin/env python
import argparse
import os
import subprocess
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", required=True, choices=["mustard", "mire2e", "mirdnn", "dnnpremir", "deepmir", "deepmirgene"])
    p.add_argument("--output-name", required=True, help="Subdirectory under results/<tool>/ to store this run")
    args = p.parse_args()

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(tools_dir)
    config_path = os.path.join(repo_root, "configs", f"{args.tool}_config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

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
