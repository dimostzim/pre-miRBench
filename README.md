# pre-miRBench

pre-miRBench builds a multispecies precursor-miRNA benchmark from MirGeneDB,
re-trains supported pre-miRNA tools, and evaluates them on controlled held-out
species and miRNA-family splits.

The current benchmark target is `mirgenedb_71`: 71 MirGeneDB species whose
precursor BED coordinates match the selected genome FASTA files. The canonical
dataset uses 200 nt RNA windows and a 1:10 positive:negative ratio.

## What the pipeline does

```text
MirGeneDB BED + genome FASTA
  -> normalize chromosome/scaffold names
  -> extract positive precursor-centered 200 nt windows
  -> mine hard negative hairpin-like 200 nt windows
  -> assign train/validation/test splits
  -> remove exact 100 nt prepared-input duplicates
  -> write dataset.csv, reports, and tool-specific inputs
  -> train Dockerized published tools
  -> evaluate on four held-out test sets
```

The 100 nt sequence is used only as the leakage-control key. Model/tool inputs
are generated from the 200 nt windows.

## Repository layout

```text
panels/mirgenedb_71/        species metadata, download URLs, provenance, supplements
pipeline/download_data.sh   download and validate MirGeneDB BEDs and genome FASTAs
pipeline/build.py            simple canonical dataset build entry point
pipeline/build_dataset.py    full dataset builder used internally by build.py
pipeline/train.py            train one supported tool with Docker
pipeline/evaluate.py         score trained tools and write metrics/plots
tools/<tool>/               Dockerfile plus train/inference adapter per tool
pre-miRBench_model/          released pre-miRBench model files
```

Supported published tools are `deepmir`, `deepmirgene`, `dnnpremir`, `mirdnn`,
`mire2e`, and `mustard`.

## Setup

Create the pipeline environment:

```bash
conda env create -f pipeline/environment.yml
conda activate premirbench
```

The Conda environment contains the software needed for downloading and building
the dataset, including Python 3.11, NumPy, SciPy, scikit-learn, PyYAML,
`bedtools`, and ViennaRNA (`RNAfold`).

Check the important command-line dependencies if needed:

```bash
python --version
RNAfold --version
bedtools --version
curl --version
```

Docker is not installed through Conda. It is only required for training and
running the published predictor containers. If you only want to build the
pre-miRBench dataset, Docker is not required.

To prepare the predictor containers for training/evaluation:

```bash
bash tools/setup_images.sh
```

## Build the dataset

### 1. Download the raw data

From the repository root:

```bash
bash pipeline/download_data.sh
```

The repository already contains the 71-species panel, source URLs, provenance,
and any small supplementary FASTA/BED files required by the panel. The large
genome FASTA files themselves are downloaded locally and are not stored in git.

By default the raw files are written to:

```text
data/raw/mirgenedb_71/
```

### 2. Build the canonical dataset

```bash
python pipeline/build.py
```

That is the normal build command. It automatically uses:

```text
data/raw/mirgenedb_71/panel.tsv
data/work/build_mirgenedb_71/
data/datasets/mirgenedb_71/
```

The default build is the canonical 1:10 benchmark and automatically chooses a
parallelism layout from the machine's available CPUs.

Useful options:

```bash
python pipeline/build.py --jobs 12
python pipeline/build.py --resume
python pipeline/build.py --species hsa,mmu,gga
python pipeline/build.py --data-dir /path/to/large/disk/premirbench-data
```

`--jobs` is the approximate total CPU budget. `--resume` reuses completed
per-species intermediate files from an interrupted build.

If `--data-dir` is used, keep the download and build locations consistent. For
example:

```bash
bash pipeline/download_data.sh /path/to/large/disk/premirbench-data/raw/mirgenedb_71
python pipeline/build.py --data-dir /path/to/large/disk/premirbench-data
```

The lower-level `pipeline/build_dataset.py` command still exists for advanced
experiments, but normal users should use `pipeline/build.py`.

## Dataset output

A completed canonical dataset directory contains:

```text
data/datasets/mirgenedb_71/
  dataset.csv
  genome.fa
  split_summary.csv
  family_split_summary.csv
  leakage_report.csv
  tool_inputs/
```

The combined genome is large; the reference build is approximately 92 GB.

## Current dataset snapshot

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

## Splits

There is one validation split and four test splits:

| split | species relation to train | family relation to train | purpose |
| --- | --- | --- | --- |
| `valid` | known species | known families | model selection |
| `test_known_species_known_family` | known species | known families | in-distribution test |
| `test_known_species_heldout_family` | known species | held-out families | family generalization |
| `test_heldout_species_known_family` | held-out species | known families | species generalization |
| `test_heldout_species_heldout_family` | held-out species | held-out families | strictest generalization test |

Final rows are globally de-duplicated by exact prepared 100 nt sequence. The same
leakage-control sequence cannot appear twice within a split, between train and
validation, between train and tests, between tests, or on both sides of the
positive/negative label. `leakage_report.csv` records the final checks.

## Train published tools

The training wrappers use Dockerized environments for the six published
predictors.

Train one tool:

```bash
python pipeline/train.py \
  --tool mirdnn \
  --run-name mirgenedb71_1to10 \
  --dataset-dir data/datasets/mirgenedb_71 \
  --output-root results/training
```

Train all tools:

```bash
for tool in deepmir deepmirgene dnnpremir mirdnn mire2e mustard; do
  python -u pipeline/train.py \
    --tool "$tool" \
    --run-name mirgenedb71_1to10 \
    --dataset-dir data/datasets/mirgenedb_71 \
    --output-root results/training
done
```

Each trained tool writes an `inference_config.yaml` next to its model artifact.

## Evaluate published tools

Evaluate all trained tools:

```bash
python pipeline/evaluate.py \
  --dataset-dir data/datasets/mirgenedb_71 \
  --training-root results/training \
  --run-name mirgenedb71_1to10 \
  --output-dir results/evaluation/mirgenedb71_1to10 \
  --resume
```

Evaluate a subset:

```bash
python pipeline/evaluate.py \
  --tools mirdnn,deepmirgene,dnnpremir \
  --dataset-dir data/datasets/mirgenedb_71 \
  --training-root results/training \
  --run-name mirgenedb71_1to10 \
  --output-dir results/evaluation/mirgenedb71_1to10 \
  --resume
```

Evaluation writes outputs such as:

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

Regenerate plots from an existing `metrics.csv` with:

```bash
python pipeline/evaluate.py \
  --output-dir results/evaluation/mirgenedb71_1to10 \
  --plot-only
```
