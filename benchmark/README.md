# Training Data Pipeline

This directory contains the data download and dataset-building code used for
tool retraining.

## Requirements

- ViennaRNA (`RNAfold`)
- Python 3
- Tool Docker images from `tools/<tool>/setup.sh` for containerized training

## Download Data

Download MirGeneDB precursor BED files and UCSC genome FASTA files with the
scripts in `benchmark/download/`.

The current diverse species panel is handled by:

```bash
bash benchmark/download/download_diverse20.sh
```

For a single species:

```bash
bash benchmark/download/download_species.sh hsa
```

See `benchmark/download/README.md` for details.

## Single-Species Dataset

Build a 1:5 positive/negative dataset from one genome and BED file:

```bash
python benchmark/train_data/build_dataset.py \
  --genome benchmark/data/chr14.fa \
  --bed benchmark/data/hsa-precursors-no-v2.bed \
  --output-dir data/train \
  --ratio 5 \
  --window 200 \
  --step 50 \
  --chr chr14
```

This produces canonical windows plus tool-specific inputs under
`data/train/tool_inputs/<tool>/`.

## Multispecies Dataset

Build train, validation, same-species held-out-chromosome test, and held-out
species test splits:

```bash
python benchmark/train_data/build_multispecies_dataset.py \
  --species-data-dir benchmark/data/species \
  --output-dir data/train_multispecies \
  --ratio 5 \
  --window 200 \
  --step 50
```

The multispecies builder prefixes contigs as `<species>__<contig>`, reserves
one positive-bearing chromosome/scaffold per training species for validation,
reserves another for same-species testing, and holds out configured species for
external testing.

## Tool Inputs

`benchmark/train_data/prepare_tool_inputs.py` writes the files consumed by
training wrappers:

- `positive.fa`, `negative.fa`
- `validation_positive.fa`, `validation_negative.fa`
- `test_chrom_positive.fa`, `test_chrom_negative.fa`
- `test_species_positive.fa`, `test_species_negative.fa`
- matching BED and metadata files where a tool needs intervals

Each tool receives the representation expected by its implementation, while all
examples originate from the shared canonical dataset.

## Train Tools

After building tool images, run a smoke training pass:

```bash
for tool in deepmir deepmirgene dnnpremir mirdnn mire2e mustard; do
  python tools/train.py --tool "$tool" --run-name smoke_gpu_1to5
done
```

The training wrapper passes `--gpus all` to Docker and uses CUDA by default.
