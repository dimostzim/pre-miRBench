# mirDNN

## Setup

### Option 1: Local Conda Environment
Installs mirDNN dependencies and clones source repository.

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
docker build -t mirdnn:latest .
```

## Download Test Data

Prepare test FASTA sequences from mirDNN repository:

```bash
./download_test_data.sh
```

## Run Inference

### Option 1: Local Conda Environment

Activate the environment:
```bash
conda activate mirdnn
```

Run inference on test data:
```bash
python inference.py \
  --input data/test.fa \
  --output results \
  --model animal
```

**Note:** The wrapper automatically runs RNAfold to generate secondary structure before prediction.

### Option 2: Docker

```bash
docker run --rm --platform linux/amd64 \
  -v $(pwd)/data:/work/data \
  -v $(pwd)/results:/work/results \
  mirdnn:latest \
  --input /work/data/test.fa \
  --output /work/results \
  --model animal
```

## Parameters

**required:**
- `--input`: FASTA file with RNA sequences.

**default:**
- `--output` (default: `"results"`): Output directory.
- `--model` (default: `"animal"`): Pre-trained model: `animal` or `plants`.
- `--seq_length` (default: `160`): Sequence length for padding/truncation.
- `--device` (default: `"cpu"`): Device for inference: `cpu` or `cuda:0`.
- `--batch_size` (default: `1024`): Batch size for prediction.

## Models

| Model | Species | Sequence Length |
|-------|---------|-----------------|
| animal | Animals (H. sapiens, C. elegans, etc.) | 160 bp |
| plants | Plants (A. thaliana, etc.) | 160 bp (320 bp for Arabidopsis) |

## Output

The tool generates:
- `input.fold`: Intermediate file with sequences + secondary structure + MFE
- `predictions.csv`: Final predictions with two columns:
  - `sequence_name`: Sequence identifier from FASTA
  - `prediction_score`: Probability score (0-1, higher = more likely pre-miRNA)

