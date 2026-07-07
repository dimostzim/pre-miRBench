# pre-miRBench

pre-miRBench builds a multispecies precursor-miRNA benchmark from MirGeneDB,
re-trains supported pre-miRNA tools, and evaluates them on controlled held-out
species and miRNA-family splits.

The current benchmark target is `mirgenedb_71`: 71 MirGeneDB species whose
precursor BED coordinates match the selected genome FASTA files. The canonical
dataset is built as 200 nt RNA windows with a 1:10 positive:negative ratio.

## What The Pipeline Does

```text
MirGeneDB BED + genome FASTA
  -> normalize chromosome/scaffold names
  -> extract positive precursor-centered 200 nt windows
  -> mine hard negative hairpin-like 200 nt windows
  -> assign train/validation/test splits
  -> remove exact 100 nt prepared-input duplicates
  -> write canonical dataset.csv and per-tool input files
  -> train Dockerized tools
  -> evaluate on four held-out test sets
```

The 100 nt sequence is used only as the leakage-control key. The model/tool
inputs are generated from the 200 nt windows.

## Repository Layout

```text
panels/mirgenedb_71/        final species panel and build snapshot notes
pipeline/download_data.sh   download MirGeneDB BED files and genome FASTA files
pipeline/build_dataset.py   build dataset.csv, genome.fa, split reports, tool inputs
pipeline/train.py           train one supported tool with Docker
pipeline/evaluate.py        score trained tools and write metrics/plots
pipeline/export_agentomics_dataset.py
                            export the benchmark for Agentomics
tools/<tool>/               Dockerfile plus train/inference adapter per tool
tests/                      unit tests and small fixtures
```

Supported tools are `deepmir`, `deepmirgene`, `dnnpremir`, `mirdnn`, `mire2e`,
and `mustard`.

## Setup

Create the pipeline environment:

```bash
conda env create -f pipeline/environment.yml
conda activate premirbench
```

Build the Docker images used for training and inference:

```bash
bash tools/setup_images.sh
```

The pipeline also needs Docker, `bedtools`, `RNAfold`, and enough disk space for
the combined genomes. Use scratch storage for full builds.

## Current Dataset Snapshot

The 2026-07-04 `mirgenedb_71` build has:

| item | value |
| --- | ---: |
| species | 71 |
| records | 77,616 |
| positives excluded as duplicate 100 nt inputs | 1,259 |
| negative:positive ratio | 10:1 in every split |
| combined genome | 92 GB |

Split counts:

| split | positives | negatives |
| --- | ---: | ---: |
| `train` | 4,765 | 47,650 |
| `valid` | 631 | 6,310 |
| `test_known_species_known_family` | 707 | 7,070 |
| `test_known_species_heldout_family` | 677 | 6,770 |
| `test_heldout_species_known_family` | 207 | 2,070 |
| `test_heldout_species_heldout_family` | 69 | 690 |

The prepared dataset is too large for git. A complete dataset directory should
look like this:

```text
mirgenedb_71/
  dataset.csv
  genome.fa
  split_summary.csv
  family_split_summary.csv
  leakage_report.csv
  run_metadata.json
  repo_diff.patch
  tool_inputs/
```

Useful environment variables:

```bash
export SCR=/SCRATCH/$USER/pre-miRBench
export DATASET=$SCR/datasets/mirgenedb_71
export TRAIN_OUT=$SCR/results/training
export EVAL_OUT=$SCR/results/evaluation/mirgenedb71_1to10
```

## Build The Dataset

Download the 71-species panel:

```bash
bash pipeline/download_data.sh "$SCR/raw/mirgenedb_71"
```

Build the canonical 1:10 dataset:

```bash
python pipeline/build_dataset.py \
  --panel "$SCR/raw/mirgenedb_71/panel.tsv" \
  --output-dir "$SCR/datasets/mirgenedb_71" \
  --work-dir "$SCR/work/build_mirgenedb_71" \
  --ratio 10 \
  --cpus 8 \
  --species-jobs 12
```

Resume a partially completed build:

```bash
python pipeline/build_dataset.py \
  --panel "$SCR/raw/mirgenedb_71/panel.tsv" \
  --output-dir "$SCR/datasets/mirgenedb_71" \
  --work-dir "$SCR/work/build_mirgenedb_71" \
  --ratio 10 \
  --cpus 8 \
  --species-jobs 12 \
  --reuse-existing
```

Important build flags:

| flag | meaning |
| --- | --- |
| `--ratio 10` | target 10 negatives per positive in every final split |
| `--window 200` | length of positive and negative windows |
| `--cpus` | RNAfold workers used inside each species job |
| `--species-jobs` | number of species processed in parallel |
| `--heldout-species` | species excluded from training for held-out-species tests |
| `--reuse-existing` | reuse completed per-species intermediate files |

With 96 CPUs, a reasonable starting point is `--species-jobs 12 --cpus 8`.

## Splits

There is one validation split and four test splits:

| split | species relation to train | family relation to train | purpose |
| --- | --- | --- | --- |
| `valid` | known species | known families | model selection only |
| `test_known_species_known_family` | known species | known families | easiest in-distribution test |
| `test_known_species_heldout_family` | known species | held-out families | family generalization |
| `test_heldout_species_known_family` | held-out species | known families | species generalization |
| `test_heldout_species_heldout_family` | held-out species | held-out families | strictest generalization test |

Final rows are globally de-duplicated by exact prepared 100 nt sequence. That
means the same leakage-control sequence cannot appear twice within a split,
between train and validation, between train and tests, between tests, or on both
sides of the positive/negative label.

`leakage_report.csv` records the final checks.

## Train Tools

Train one tool:

```bash
python pipeline/train.py \
  --tool mirdnn \
  --run-name mirgenedb71_1to10 \
  --dataset-dir "$DATASET" \
  --output-root "$TRAIN_OUT"
```

Train all tools:

```bash
for tool in deepmir deepmirgene dnnpremir mirdnn mire2e mustard; do
  PYTHONUNBUFFERED=1 python -u pipeline/train.py \
    --tool "$tool" \
    --run-name mirgenedb71_1to10 \
    --dataset-dir "$DATASET" \
    --output-root "$TRAIN_OUT"
done
```

Each trained tool writes an `inference_config.yaml` next to its model artifact:

```text
$TRAIN_OUT/<tool>/mirgenedb71_1to10/
```

## Evaluate Tools

Evaluate all trained tools:

```bash
python pipeline/evaluate.py \
  --dataset-dir "$DATASET" \
  --training-root "$TRAIN_OUT" \
  --run-name mirgenedb71_1to10 \
  --output-dir "$EVAL_OUT" \
  --resume
```

Evaluate a subset:

```bash
python pipeline/evaluate.py \
  --tools mirdnn,deepmirgene,dnnpremir \
  --dataset-dir "$DATASET" \
  --training-root "$TRAIN_OUT" \
  --run-name mirgenedb71_1to10 \
  --output-dir "$EVAL_OUT" \
  --resume
```

Evaluation writes:

```text
predictions.csv
metrics.csv
metrics_by_species.csv
run.log.txt
auprc_by_tool.svg
auprc_by_tool.png
raw/
inputs/
```

Regenerate only the plots from an existing `metrics.csv`:

```bash
python pipeline/evaluate.py \
  --output-dir "$EVAL_OUT" \
  --plot-only
```

## Agentomics Export

Export the benchmark to Agentomics format:

```bash
python pipeline/export_agentomics_dataset.py --overwrite
```

Default output:

```text
/home/dtzim01/agentomics-ml/datasets/premirbench_mirgenedb71/
/home/dtzim01/agentomics-ml/test_datasets/premirbench_mirgenedb71/
```

Public Agentomics splits are `train` and `validation`. The four benchmark test
splits are hidden under `test_datasets/`.

Each split contains:

```text
labels.csv
input/
  samples.tsv
  phact_premirna_positions.tsv
  phact_premirna_index.tsv
```

`samples.tsv` intentionally contains only modeling inputs:

```text
id
species
sequence_rna
structure
mfe
```

It does not include coordinates, MirGeneDB IDs, family IDs, split reasons,
negative-mining scores, or internal leakage-control sequences. Those are
provenance fields and can leak label or benchmark construction details.

The PHACT files are optional global human precursor-miRNA reference tables. They
are not keyed to every benchmark sample and should not be treated as labels.

## Useful Checks

Check a copied dataset:

```bash
ls -lh "$DATASET"/dataset.csv "$DATASET"/genome.fa "$DATASET"/leakage_report.csv
ls -lh "$DATASET"/tool_inputs
```

Check trained model artifacts:

```bash
find "$TRAIN_OUT" -name inference_config.yaml | sort
find "$TRAIN_OUT" -type f | grep -E '/(model\.h5|new_test\.hdf5|CNN_model\.h5|model\.pmt|predictor\.pkl|CNNonRaw\.hdf5)$' | sort
```

Run unit tests:

```bash
python -m unittest
```
