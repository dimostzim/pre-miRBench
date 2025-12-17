# Benchmarking of Pre-miRNA Prediction Tools


Unified wrapper repository with consistent setup and inference interfaces for multiple pre-mirna prediction tools.

## Available Tools

- **mustard** - MuStARD (Multi-Scale Target RNA Detector)
- **mire2e** - miRe2e (End-to-end Transformer model)
- **mirdnn** - mirDNN (Hybrid CNN/RNN with structure)
- **dnnpremir** - dnnPreMiR (CNN-based classifier)
- **deepmir** - DeepMir (CNN-based predictor)
- **deepmirgene** - deepMiRGene (RNN with attention for pre-miRNA)

## Setup

Prerequisites:
- Conda (Miniconda/Anaconda): https://docs.anaconda.com/miniconda/install/
- Docker (for `--docker` mode): https://www.docker.com/products/docker-desktop/ (ensure the daemon is running)
- PyYAML: `pip install pyyaml` (required for unified interface)

### Conda Setup

Creates a conda env `{tool}` with dependencies and installs the tool.

```bash
./setup.sh --tool {tool}
```

### Docker Setup

Builds a Docker image `{tool}:latest` so you can run via `--docker` without installing dependencies on the host.

```bash
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
