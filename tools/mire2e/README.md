# miRe2e

## Setup

### Option 1: Local Conda Environment
Installs miRe2e dependencies and package.

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
docker build -t mire2e:latest .
```

## Download Test Data

Download example FASTA sequences:

```bash
./download_test_data.sh
```

## Run Inference

### Option 1: Local Conda Environment

Activate the environment:
```bash
conda activate mire2e
```

Run inference on test data:
```bash
python inference.py \
  --input data/examples/chr19_13836201_13836660_true.fa \
  --output results
```

**Note:** By default uses CPU. For GPU, add `--device cuda`.

### Option 2: Docker

```bash
docker run --rm --platform linux/amd64 \
  -v $(pwd)/data:/work/data \
  -v $(pwd)/results:/work/results \
  mire2e:latest \
  --input /work/data/examples/chr19_13836201_13836660_true.fa \
  --output /work/results
```

## Parameters

**required:**
- `--input`: FASTA file with RNA sequences.

**default:**
- `--output` (default: `"results"`): Output directory.
- `--device` (default: `"cpu"`): Device for inference: `cpu` or `cuda`.
- `--pretrained` (default: `"hsa"`): Pretrained model: `hsa` (H. sapiens) or `animals`.
- `--length` (default: `100`): Window length in bp.
- `--step` (default: `20`): Step size in bp for sliding window.
- `--batch_size` (default: `4096`): Batch size for prediction.

## Models

| Model | Species |
|-------|---------|
| hsa | H. sapiens |
| animals | Animals (excluding H. sapiens) |

## Data

Test data is available in the GitHub repository:
- Examples: https://github.com/sinc-lab/miRe2e/tree/master/examples
