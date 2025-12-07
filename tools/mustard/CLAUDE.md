# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a wrapper repository for MuStARD (Multi-Scale Target RNA Detector), a deep learning tool for predicting miRNA precursors and other RNA-binding protein sites. The repository provides simplified setup and inference scripts that wrap the original Perl-based MuStARD pipeline.

## Setup and Environment

### Local Setup
```bash
./setup.sh
```
Downloads dependencies (Python 2.7, Perl, ViennaRNA, bedtools, TensorFlow 1.15, Keras 2.3.1), clones the original MuStARD source from GitLab, and downloads 7 pre-trained CNN models.

### Docker Setup (CPU)
```bash
./setup.sh --docker
```
Builds a Docker image with all dependencies and models pre-installed (CPU-only TensorFlow).

### Docker Setup (GPU)
```bash
./setup.sh --docker --gpu
```
Builds a Docker image with TensorFlow GPU support. Requires:
- NVIDIA GPU with CUDA 10.0+ support
- nvidia-docker runtime installed
- Linux host (GPU not available on macOS)

Run with: `docker run --gpus all ...`

### Test Data
```bash
./download_test_data.sh
```
Downloads chr14 test data including: BED files, reference genome (hg38 chr14), and PhyloP conservation scores.

## Running Inference

### Conda Environment
```bash
conda activate mustard
python inference.py --input <bed> --genome <fasta> --cons <consDir> --output <outdir> --model MuStARD-mirSFC-U
```

### Docker (CPU)
```bash
docker run --rm --platform linux/amd64 \
  -v $(pwd)/data:/work/data \
  -v $(pwd)/results:/work/results \
  mustard:latest \
  --input /work/data/input.bed \
  --genome /work/data/genome.fa \
  --cons /work/data \
  --output /work/results \
  --model MuStARD-mirSFC-U
```

### Docker (GPU)
```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/work/data \
  -v $(pwd)/results:/work/results \
  mustard:latest \
  --input /work/data/input.bed \
  --genome /work/data/genome.fa \
  --cons /work/data \
  --output /work/results \
  --model MuStARD-mirSFC-U
```
Note: GPU image must be built with `./setup.sh --docker --gpu`

## Architecture

### Wrapper Layer
- `inference.py`: Python wrapper that translates user-friendly arguments into the complex Perl script invocation
- Maps model names to input modes (sequence/RNAfold/conservation combinations)
- Handles path resolution and directory creation
- Invokes `mustard_src/MuStARD.pl predict` with appropriate parameters

### MuStARD Source
- Cloned from https://gitlab.com/RBP_Bioinformatics/mustard.git into `mustard_src/`
- Core Perl pipeline (`MuStARD.pl`) handles:
  - Feature extraction (sequence, RNA folding, conservation)
  - Sliding window generation
  - Keras model inference
  - Result aggregation

### Pre-trained Models
Located in `data/models/<model_name>/CNNonRaw.hdf5`:
- MuStARD-mirS: Sequence only
- MuStARD-mirF: RNAfold only
- MuStARD-mirSF: Sequence + RNAfold
- MuStARD-mirSC: Sequence + Conservation
- MuStARD-mirFC: RNAfold + Conservation
- MuStARD-mirSFC: Sequence + RNAfold + Conservation
- MuStARD-mirSFC-U: Best performing model (all inputs)

Models requiring conservation (suffix 'C') need `--cons` parameter pointing to PhyloP conservation files.

## Legacy Dependencies

This project uses Python 2.7, TensorFlow 1.15, and Keras 2.3.1 due to the original MuStARD implementation. The conda environment handles version pinning. On macOS ARM, some Perl modules (e.g., perl-io-gzip) may not be available from standard channels.

## Input/Output Format

**Input**: BED format file with genomic coordinates of regions to score
**Output**: Directory containing:
- Prediction scores for sliding windows
- Aggregated results
- R plots (via ggplot2)

For whole-genome scans, use full hg38 genome FASTA and all phyloP files from UCSC (see README.md for URLs).
