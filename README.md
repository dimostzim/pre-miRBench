# pre-miRBench

Pipeline for building a multispecies pre-miRNA benchmark, retraining supported
tools, and evaluating them on controlled held-out splits.

The current benchmark is based on the `mirgenedb_71` panel: 71 MirGeneDB
species with matched genome FASTA files and verified precursor coordinates.

## Repository Contents

- `panels/mirgenedb_71/` - final species panel and small supplement addenda
- `pipeline/download_data.sh` - download BED/FASTA data for the panel
- `pipeline/build_dataset.py` - build the canonical dataset and tool inputs
- `pipeline/train.py` - train one Dockerized tool
- `pipeline/evaluate.py` - evaluate trained tools on the test splits
- `tools/<tool>/` - Dockerfiles plus training/inference adapters
- `tests/fixtures/` - small smoke-test fixtures

## Setup

Create the pipeline environment:

```bash
conda env create -f pipeline/environment.yml
conda activate premirbench
```

Build the tool Docker images:

```bash
bash tools/setup_images.sh
```

## Provided Dataset

The prepared dataset can be provided separately from git because it is large.
If you already have it, place it anywhere convenient and point commands to it
with `--dataset-dir`.

Expected layout:

```text
mirgenedb_71/
  dataset.csv
  genome.fa
  split_summary.csv
  family_split_summary.csv
  leakage_report.csv
  tool_inputs/
```

Example:

```bash
export SCR=/SCRATCH/$USER/pre-miRBench
export DATASET=$SCR/datasets/mirgenedb_71
export TRAIN_OUT=$SCR/results/training
```

## Rebuild Dataset

Download the panel:

```bash
bash pipeline/download_data.sh data/raw/mirgenedb_71
```

Build the 1:10 dataset:

```bash
python pipeline/build_dataset.py \
  --panel data/raw/mirgenedb_71/panel.tsv \
  --output-dir data/datasets/mirgenedb_71 \
  --work-dir data/work/build_mirgenedb_71 \
  --ratio 10 \
  --cpus 8 \
  --species-jobs 12
```

For scratch storage:

```bash
export SCR=/SCRATCH/$USER/pre-miRBench

bash pipeline/download_data.sh "$SCR/raw/mirgenedb_71"

python pipeline/build_dataset.py \
  --panel "$SCR/raw/mirgenedb_71/panel.tsv" \
  --output-dir "$SCR/datasets/mirgenedb_71" \
  --work-dir "$SCR/work/build_mirgenedb_71" \
  --ratio 10 \
  --cpus 8 \
  --species-jobs 12
```

Useful build flags:

- `--ratio 10` means 10 negatives per positive.
- `--cpus` controls RNAfold workers per species.
- `--species-jobs` controls how many species run in parallel.
- `--reuse-existing` reuses completed intermediate files.
- `--heldout-species` controls which species are absent from train.

## Splits

The builder creates one validation split and four test splits:

| Split | Meaning |
| --- | --- |
| `valid` | Same species and same miRNA families as train. Used only for model selection. |
| `test_known_species_known_family` | Same species and same miRNA families as train. |
| `test_known_species_heldout_family` | Same species as train, held-out miRNA families. |
| `test_heldout_species_known_family` | Held-out species, same miRNA families as train. |
| `test_heldout_species_heldout_family` | Held-out species and held-out miRNA families. |

Final rows are globally de-duplicated by exact prepared 100 nt sequence, so the
same prepared input sequence cannot appear in multiple splits or labels.
`leakage_report.csv` records the final checks.

## Train

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

Each trained tool writes an `inference_config.yaml` next to its model artifact.

## Evaluate

Evaluate all trained tools:

```bash
python pipeline/evaluate.py \
  --dataset-dir "$DATASET" \
  --training-root "$TRAIN_OUT" \
  --run-name mirgenedb71_1to10 \
  --output-dir "$SCR/results/evaluation/mirgenedb71_1to10"
```

Evaluation writes predictions, metrics, per-species metrics, logs, and AUPRC
bar plots for the four test splits.
