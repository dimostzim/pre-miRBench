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

**required:**
- `--input`: Input FASTA file.

**default:**
- `--output` (default: `"results"`): Output directory.
- `--model` (optional): Custom `.hdf5` weight file to use instead of bundled model.

## Model

| Model | Type | Backend | Location |
|-------|------|---------|----------|
| new_test.hdf5 | LSTM RNN | Theano | Bundled in deepmirgene_src/inference/model/ |

The bundled model is automatically used unless a custom model is specified via `--model`.

## Output

The tool generates:
- `predictions.txt`: Predictions with sequence IDs and classification results
