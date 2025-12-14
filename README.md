# Pre-miRNA Prediction Tools

Unified wrapper repository with consistent setup and inference interfaces for multiple prediction tools.

## Setup

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
python inference.py --tool {tool}
```

Automatically activates the conda environment.

### Docker Mode

```bash
python inference.py --tool {tool} --docker
```

### Results

Results are saved to `results/{tool}/`.

## Requirements

- **Conda mode**: miniconda/anaconda
- **Docker mode**: Docker Desktop

## Available Tools

- **mustard** - MuStARD (Multi-Scale Target RNA Detector)
- **mire2e** - miRe2e (End-to-end Transformer model)
