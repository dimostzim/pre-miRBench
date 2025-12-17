# deepMiRGene

## Setup

### Option 1: Local Conda Environment
Installs dependencies and clones the upstream deepMiRGene repo (includes pretrained weights).

```bash
./setup.sh
```

### Option 2: Docker
Build an image that already contains the environment and source.

```bash
docker build -t deepmirgene:latest .
```

## Download Test Data

Grab the example FASTA file used in the original repository:

```bash
./download_test_data.sh
```

## Run Inference

### Option 1: Local Conda Environment

Activate the environment:
```bash
conda activate deepmirgene
```

Run on the example FASTA:
```bash
python inference.py \
  --input data/examples.fa \
  --output results
```

### Option 2: Docker

```bash
docker run --rm --platform linux/amd64 \
  -v $(pwd)/data:/work/data \
  -v $(pwd)/results:/work/results \
  deepmirgene:latest \
  --input /work/data/examples.fa \
  --output /work/results
```

## Parameters

- `--input` (required): Input FASTA file.
- `--output` (default: `results`): Directory to store `predictions.txt`.
- `--model` (optional): Custom `.hdf5` weight file to stage as `model/new_test.hdf5` for inference.
