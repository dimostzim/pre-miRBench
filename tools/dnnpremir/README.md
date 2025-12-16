# dnnPreMiR

## Setup

### Option 1: Local Conda Environment
Installs dnnPreMiR dependencies and clones source repository.

Install conda: https://docs.anaconda.com/miniconda/install/

Then run:

```bash
./setup.sh
```

### Option 2: Docker

Install Docker Desktop: https://www.docker.com/products/docker-desktop/

Ensure Docker is running before building image or running containers.

Then run:

```bash
docker build -t dnnpremir:latest .
```

## Download Test Data

Prepare test FASTA sequences from dnnPreMiR repository:

```bash
./download_test_data.sh
```

## Run Inference

### Option 1: Local Conda Environment

Activate the environment:
```bash
conda activate dnnpremir
```

Run inference on test data:
```bash
python inference.py \
  --input data/test.fa \
  --output results
```

**Note:** The wrapper automatically handles running RNAfold for secondary structure prediction.

### Option 2: Docker

```bash
docker run --rm --platform linux/amd64 \
  -v $(pwd)/data:/work/data \
  -v $(pwd)/results:/work/results \
  dnnpremir:latest \
  --input /work/data/test.fa \
  --output /work/results
```

## Parameters

**required:**
- `--input`: FASTA file with RNA sequences.

**default:**
- `--output` (default: `"results"`): Output directory.
- `--seq_length` (default: `180`): Sequence length (fixed at 180 nt, automatically truncated/padded).

## Model

| Model | Type | Sequence Length |
|-------|------|-----------------|
| CNN_model.h5 | Convolutional Neural Network | 180 nt (fixed) |

The model is trained on human pre-miRNAs from miRBase release 22 and pseudo pre-miRNAs from human coding regions.

## Output

The tool generates:
- `predictions.txt`: Predictions with sequence names and classification (True/False for pre-miRNA)

**Important:** Input sequences longer than 180 nt are truncated; shorter sequences are padded to 180 nt.
