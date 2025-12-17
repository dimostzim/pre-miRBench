# MuStARD

## Setup

### Option 1: Local Conda Environment
Installs MuStARD dependencies, clones source + pretrained models.

Install conda: https://docs.anaconda.com/miniconda/install/

Then run:

```bash
./setup.sh
```

### Option 2: Docker

**Note:** Docker is required for ARM cpus since Python 2.7 is not available natively.

Install Docker Desktop: https://www.docker.com/products/docker-desktop/

Ensure Docker is running before building image or running containers.

Then run:

```bash
docker build -t mustard:latest .
```

## Download Test Data

```bash
./download_test_data.sh
```

## Run Inference

### Option 1: Local Conda Environment

Activate the environment:
```bash
conda activate mustard
```

Run inference on test data using the best model:
```bash
python inference.py \
  --targetIntervals data/test_loci/hsa.hairpin.slop5k.bothStrands.chr14.bed \
  --genome data/chr14.fa \
  --consDir data \
  --chromList chr14
```

**Note:** The conda environment uses `tensorflow-gpu==1.15`, which automatically uses GPU if available on Linux systems, or falls back to CPU if no GPU is detected.

### Option 2: Docker

```bash
docker run --rm --platform linux/amd64 \
  -v $(pwd)/data:/work/data \
  -v $(pwd)/results:/work/results \
  mustard:latest \
  --targetIntervals /work/data/test_loci/hsa.hairpin.slop5k.bothStrands.chr14.bed \
  --genome /work/data/chr14.fa \
  --consDir /work/data \
  --chromList chr14 \
  --dir /work/results
```

## Parameters

All parameters of MuStARD.pl predict interface:

**required:**
- `--targetIntervals`: BED formatted file with intervals for prediction.
- `--genome`: Genome FASTA file.
- `--consDir`: Directory with PhyloP conservation files (for Conservation models).

**default:**
- `--chromList` (default: `"all"`): List of chromosomes to scan (e.g., `"chr14"` or `"chr1,chr2"` or `"all"`).
- `--dir` (default: `"results"`): Working directory for results.
- `--model` (default: `"MuStARD-mirSFC-U"`): Model name.
- `--classNum` (default: `2`): Number of classes.
- `--modelType` (default: `"CNN"`): Model type: CNN, CAE, or CVAE.

**optional:**
- `--modelDirName` (default: `"results"`): Name of model subdirectory for results.
- `--intermDir` (default: `"same"`): Directory for pre-processed files (defaults to same as `--dir`).
- `--winSize` (default: `100`): Window size in bp.
- `--step` (default: `5`): Step size in bp for sliding window.
- `--staticPredFlag` (default: `0`): `0` = sliding-window scanning, `1` = static prediction (score intervals as-is).
- `--inputMode` (default: `"sequence,RNAfold,conservation"`): Input mode: sequence, RNAfold, conservation (comma-separated). Default matches best model (MuStARD-mirSFC-U).
- `--threads` (default: `10`): Number of threads for pre-processing.


## Models

| Model | Inputs |
|-------|--------|
| MuStARD-mirS | S |
| MuStARD-mirF | F |
| MuStARD-mirSF | S+F |
| MuStARD-mirSC | S+C |
| MuStARD-mirFC | F+C |
| MuStARD-mirSFC | S+F+C |
| MuStARD-mirSFC-U | S+F+C |

S=Sequence, F=RNAfold, C=Conservation

## Output

The tool generates:
- Prediction scores for sliding windows across genomic regions
- Aggregated peaks identifying high-confidence pre-miRNA candidates
- R plots visualizing prediction results

## Data
Full genome:
- Genome: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
- PhyloP: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/
