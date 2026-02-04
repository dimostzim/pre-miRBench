# Benchmark Pipeline

Pre-miRNA prediction benchmark: RNA folding, balanced dataset creation, and model training.

## Requirements

- ViennaRNA
- bedtools
- Python packages: `numpy`, `matplotlib`, `scikit-learn`, `scipy`

## Setup

```bash
conda env create -f environment.yml
conda activate benchmark
```

## Download Data

See `download/README.md` for data download scripts.

## Folding Pipeline

Runs RNAfold on sliding windows (default: 200bp, step 50bp) across sequences.

```bash
python fold/run_folding.py --input data/..
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
python fold/find_mirna_windows.py \
  --csv output/fold_output/results.csv \
  --bed download/data/cel-precursors-no-v2.bed \
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
python make_negative_set/sample_negatives.py \
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

## Train Baseline Models

```bash
python models/train_baselines.py \
  --input output/sample_negatives_output/balanced.csv
```

**Parameters:**
- `--input`: Balanced dataset CSV
- `--json`: Output JSON file (default: output/models_output/baseline_metrics.json)
- `--plot`: Output plot file (default: {json}.png)
- `--test_size`: Test split ratio (default: 0.1)
- `--random_state`: Random seed (default: 42)
- `--C`: Regularization strength for LR (default: 1.0)
- `--solver`: Solver algorithm for LR (default: lbfgs)
- `--n_trees`: Number of trees for RF/GB (default: 100)
- `--svm_C`: SVM regularization (default: 1.0)
- `--svm_kernel`: SVM kernel - rbf/linear/poly (default: rbf)

**Baselines:**
- `random`: Random classifier (should be ~0.5)
- `lr_dinuc`: Logistic regression on dinucleotide features
- `rf_dinuc`: Random forest on dinucleotide features
- `svm_dinuc`: SVM on dinucleotide features
- `gb_dinuc`: Gradient boosting on dinucleotide features

**Outputs:**
- `output/models_output/baseline_metrics.json`: JSON file with metrics
- `output/models_output/baseline_metrics.png`: Precision-recall curves plot
