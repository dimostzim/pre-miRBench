# Benchmark Pipeline

Pre-miRNA prediction benchmark: RNA folding, dataset creation, tool-specific input preparation, and evaluation.

## Requirements

- ViennaRNA
- bedtools
- Python packages: `numpy`, `matplotlib`, `scikit-learn`, `scipy`

## Setup

Run the following commands from the repository root unless noted otherwise.

```bash
conda env create -f benchmark/environment.yml
conda activate benchmark
```

Build the tool Docker images once before running the tool benchmark:

```bash
cd tools
./setup.sh --tool deepmir
./setup.sh --tool deepmirgene
./setup.sh --tool dnnpremir
./setup.sh --tool mirdnn
./setup.sh --tool mire2e
./setup.sh --tool mustard
cd ..
```

## Balanced Benchmark

The current end-to-end tool benchmark is the collapsed balanced dataset:

- positives: one representative 200 nt window per pre-miRNA
- negatives: matched 1:1 negatives
- implementation: `benchmark/balanced_benchmark/`
- source dataset: `benchmark/balanced_benchmark/datasets/balanced_benchmark.csv`

### Full Pipeline

If `benchmark/balanced_benchmark/datasets/balanced_benchmark.csv` is missing, rebuild it once:

```bash
python benchmark/balanced_benchmark/build_dataset.py
```

Prepare tool-specific inputs:

```bash
python benchmark/balanced_benchmark/prepare_inputs.py
```

Run all tool wrappers:

```bash
bash benchmark/balanced_benchmark/run_tools.sh
```

Normalize outputs and compute shared label-based metrics:

```bash
python benchmark/balanced_benchmark/evaluate_outputs.py
```

### Outputs

- prepared inputs: `benchmark/prepared_inputs/balanced_benchmark/`
- raw tool outputs: `results/{tool}/balanced_benchmark/`
- normalized per-tool outputs: `benchmark/evaluated/balanced_benchmark/{tool}.csv`
- summary metrics: `benchmark/evaluated/balanced_benchmark/metrics.csv`

### Tool-Specific Prepared Inputs

- `deepmir`: original 200 nt sequence FASTA
- `deepmirgene`: original 200 nt sequence FASTA
- `dnnpremir`: 180 nt crop
- `mirdnn`: 160 nt crop
- `mire2e`: 100 nt target-aware crop
- `mustard`: 100 nt target-aware BED intervals, static mode

The `mire2e` and `mustard` 100 nt inputs are shifted when needed so every positive still fully contains the target pre-miRNA.

### Evaluation Notes

- metrics are restricted to values available for every tool: `tp`, `fp`, `tn`, `fn`, `precision`, `recall`, `specificity`, `accuracy`, `f1`, `mcc`
- MuStARD is normalized using `class_0` as the positive class, matching its source training/evaluation code
- this benchmark is a candidate-level collapsed balanced benchmark, not the full scan benchmark

## Scan Benchmark

The scan benchmark searches full `chr14` and evaluates recovered precursor loci:

- implementation: `benchmark/scan_benchmark/`
- search space: `benchmark/data/chr14.fa`
- truth loci: `benchmark/data/hsa-precursors-no-v2.bed` filtered to `chr14`
- default result prefix: `scan_chr14`

This is one shared scan benchmark for all tools:

- `mire2e` and `mustard` use native scan mode
- `deepmir`, `deepmirgene`, `dnnpremir`, and `mirdnn` use externally generated sliding windows over `chr14`

### Full Pipeline

Prepare chunked scan inputs:

```bash
python benchmark/scan_benchmark/prepare_inputs.py
```

Run all tool wrappers chunk by chunk:

```bash
bash benchmark/scan_benchmark/run_tools.sh
```

Normalize raw scan outputs, merge positive windows into loci, and compute locus-level metrics:

```bash
python benchmark/scan_benchmark/evaluate_outputs.py
```

### Outputs

- prepared inputs: `benchmark/prepared_inputs/scan_benchmark/`
- raw tool outputs: `results/{tool}/scan_chr14/{chunk_id}/`
- normalized window outputs: `benchmark/evaluated/scan_benchmark/{tool}.windows.csv`
- merged locus outputs: `benchmark/evaluated/scan_benchmark/{tool}.loci.csv`
- summary metrics: `benchmark/evaluated/scan_benchmark/metrics.csv`

### Scan Preparation

- chunk size defaults to `1,000,000` bp with `200` bp overlap for native scan tools
- non-native scan windows use a global stride of `50` bp by default
- external window sizes:
  - `deepmir`: `200`
  - `deepmirgene`: `200`
  - `dnnpremir`: `180`
  - `mirdnn`: `160`

This benchmark is substantially heavier than the balanced benchmark. Start with `--tools` if you want to run only one or two tools first.

### Scan Evaluation Notes

- score threshold is fixed at `0.5`
- positive windows are merged into loci when they overlap on the same strand
- a predicted locus counts as a hit when it overlaps at least `50%` of a truth locus
- each truth locus can be matched at most once
- metrics are locus-level:
  - `predicted_loci`
  - `matched_truth_loci`
  - `false_positive_loci`
  - `precision_locus`
  - `locus_recall`
  - `fp_per_mb`

## Download Data

See `download/README.md` for data download scripts.

## Folding Pipeline

Runs RNAfold on sliding windows (default: 200bp, step 50bp) across sequences.

```bash
python benchmark/fold/run_folding.py --input benchmark/data/chr14.fa
```

**Parameters:**
- `--input`: Input FASTA file
- `--output`: Output directory (default: output/fold_output)
- `--window`: Window size (default: 200)
- `--step`: Step size (default: 50)
- `--chr`: Chromosomes to process (default: all)
- `--cpus`: Number of CPUs (default: 8)
- `--dna`: Keep DNA bases (default: converts T→U)
- `--single_strand`: Only forward strand (default: both strands)
- `--max_repeat_frac`: Max repeat fraction (default: 0.1)

**Outputs:**
- `output/fold_output/windows.fa`: Sliding windows FASTA
- `output/fold_output/windows.fold`: RNAfold output
- `output/fold_output/results.csv`: Window data
- `output/fold_output/mfe_kde.png`: MFE distribution plot

## Find miRNA-Containing Windows

```bash
python benchmark/fold/find_mirna_windows.py \
  --csv output/fold_output/results.csv \
  --bed benchmark/data/hsa-precursors-no-v2.bed \
  --output output/fold_output/positives.csv \
  --output_collapsed output/fold_output/positives_collapsed.csv \
  --summary output/fold_output/summary.txt \
  --plot output/fold_output/mirna_mfe.png
```

**Notes:**
- `positives.csv` contains **all** windows that fully contain a pre-miRNA (multiple windows per miRNA).
- `positives_collapsed.csv` contains **one window per miRNA**, chosen by closest window center to the miRNA midpoint (ties broken by lower MFE, then earlier start).

## Sample Negatives & Build Datasets

This step produces **four datasets**:
1) `balanced.csv`: all positives + sampled negatives (1:1, overlapping)  
2) `imbalanced.csv`: all positives + all negatives (overlapping)  
3) `balanced_collapsed.csv`: collapsed positives (one per miRNA) + sampled **non-overlapping** negatives (1:1)  
4) `imbalanced_collapsed.csv`: collapsed positives + all **non-overlapping** negatives  

Negative sampling matches MFE, dinucleotide, and structure distributions of positives **only for the 1:1 datasets** (`balanced.csv` and `balanced_collapsed.csv`). The imbalanced datasets use all negatives (no matching).

```bash
python benchmark/make_negative_set/sample_negatives.py \
  --positives output/fold_output/positives.csv \
  --positives_collapsed output/fold_output/positives_collapsed.csv \
  --all_windows output/fold_output/results.csv
```

**Parameters:**
- `--positives`: CSV with positive windows
- `--positives_collapsed`: Optional CSV with one window per miRNA (collapsed positives)
- `--all_windows`: CSV with all windows
- `--balanced`: Output CSV with combined dataset (default: output/sample_negatives_output/balanced.csv)
- `--imbalanced`: Output CSV with combined dataset using all negatives (default: output/sample_negatives_output/imbalanced.csv)
- `--balanced_collapsed`: Output CSV for collapsed positives + non-overlapping negatives (default: output/sample_negatives_output/balanced_collapsed.csv)
- `--imbalanced_collapsed`: Output CSV for collapsed positives + all non-overlapping negatives (default: output/sample_negatives_output/imbalanced_collapsed.csv)
- `--plot_dir`: Directory for plots (default: output/sample_negatives_output/plots)
- `--match_strand`: Match strand distribution (default: on)
- `--match_chr`: Match chromosome distribution (default: on)
- `--seed`: Random seed (default: 42)
- `--nonoverlap_seed`: Seed for non-overlapping negative selection (default: 42)
- `--mfe_weight`: Weight for MFE feature (default: 2.0)
- `--dinuc_weight`: Weight for dinucleotide features (default: 1.0)
- `--struct_weight`: Weight for structure features (default: 1.0)
- `--complexity_weight`: Weight for complexity feature (default: 1.0)

**Matching features (22 total):**
- MFE (1)
- Dinucleotide frequencies (16)
- Structure features (4): stem_length, loop_size, bulge_count, paired_fraction
- Sequence complexity (1): Shannon entropy

**Outputs:**
- `output/sample_negatives_output/balanced.csv`: All positives + sampled negatives (1:1, overlapping)
- `output/sample_negatives_output/imbalanced.csv`: All positives + all negatives (overlapping)
- `output/sample_negatives_output/balanced_collapsed.csv`: Collapsed positives + sampled non-overlapping negatives (1:1)
- `output/sample_negatives_output/imbalanced_collapsed.csv`: Collapsed positives + all non-overlapping negatives
- `output/sample_negatives_output/plots/`: Comparison plots
