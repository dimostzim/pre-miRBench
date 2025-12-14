# Pre-miRNA Prediction Tools

Unified wrapper repository with consistent setup and inference interfaces for multiple prediction tools.

## Available Tools

- **mustard** - MuStARD (Multi-Scale Target RNA Detector)
- **mire2e** - miRe2e (End-to-end Transformer model)

## Setup

Prerequisites:
- Conda (Miniconda/Anaconda): https://docs.anaconda.com/miniconda/install/
- Docker (for `--docker` mode): https://www.docker.com/products/docker-desktop/ (ensure the daemon is running)

Install a tool using conda or Docker:

```bash
# Conda environment
./setup.sh --tool {tool}

# Docker image
./setup.sh --tool {tool} --docker
```

Download test data:

```bash
cd tools/{tool}
./download_test_data.sh
```

## Configuration

Edit `configs/{tool}_config.yaml` to set parameters. Each config file shows required and optional parameters with defaults.

## Run Inference

### Conda Mode

```bash
python inference.py --tool {tool} --output-name my_run
```

Automatically activates the conda environment.

### Docker Mode

```bash
python inference.py --tool {tool} --docker --output-name my_run
```

### Results

Results are saved to `results/{tool}/{output-name}/`.

## Requirements

- **Conda mode**: miniconda/anaconda
- **Docker mode**: Docker Desktop
