# RNA Folding Pipeline

Runs RNAfold on sliding windows (default: 200bp, step 50bp) across sequences and generates MFE distribution plot.

## Requirements

- ViennaRNA
- bedtools
- Python packages: `numpy`, `matplotlib`, `scikit-learn`, `scipy`

## Setup

```bash
conda env create -f environment.yml
conda activate folding
```

## Download Data

```bash
./download_data.sh
```

Downloads and prepares:
- `data/ce11.fa` - C. elegans genome
- `data/ce11.masked.fa` - Repeat-masked genome (repeats replaced with N)
- `data/ce11.gtf` - Gene annotation from Ensembl (for genomic region filtering)
- `data/cel-precursors-no-v2.bed` - 138 precursor coordinates (filtered from MirGeneDB)

## Usage

```bash
python run_folding.py \
  --input data/ce11.masked.fa \
  --output folding_output/ \
  --cpus 16
```

**Parameters:**
- `--input`: Input FASTA file (use repeat-masked for filtering)
- `--output`: Output directory
- `--window`: Window size (default: 200)
- `--step`: Step size (default: 50)
- `--chr`: Comma-separated list of chromosomes to process (default: all)
- `--cpus`: Number of CPUs for parallel folding (default: 8)
- `--dna`: Keep DNA bases (default: converts T→U)
- `--both_strands`: Generate windows for both strands (default: on)
- `--single_strand`: Only process forward strand
- `--max_repeat_frac`: Skip windows with repeat fraction above threshold (default: 0.1 = skip >10% repeats)

**Outputs:**
- `windows.fa`: Sliding windows FASTA
- `windows.fold`: RNAfold output (sequence + structure + MFE)
- `results.csv`: Window data (window_id, chrom, start, end, strand, sequence, structure, mfe)
- `mfe_kde.png`: KDE plot with mean and mean±SD lines

**Example (single chromosome, using defaults):**
```bash
python run_folding.py --input data/ce11.masked.fa --output output/ --chr chrI
```

## Find miRNA-Containing Windows

Find all windows that fully contain miRNA precursors from a BED file.

```bash
python scripts/find_mirna_windows.py \
  --csv folding_output/results.csv \
  --bed data/cel-precursors-no-v2.bed \
  --output folding_output/mirna_windows.csv \
  --summary folding_output/summary.txt \
  --plot folding_output/mirna_mfe_kde.png
```

**Parameters:**
- `--csv`: Input results.csv from folding
- `--bed`: BED file with miRNA coordinates
- `--output`: Output CSV with windows containing miRNAs
- `--summary`: Summary statistics (windows per miRNA)
- `--plot`: MFE distribution plot for miRNA windows

**Outputs:**
- `mirna_windows.csv`: Windows containing miRNAs (with contained_mirnas, num_mirnas columns)
- `summary.txt`: Statistics showing windows per miRNA
- `mirna_mfe_kde.png`: MFE distribution for miRNA-containing windows

## Sample Balanced Negatives

Sample negative windows matching MFE, dinucleotide, and structure distributions of positives (1:1 ratio) using nearest-neighbor matching.

```bash
python scripts/sample_negatives.py --positives output/positives.csv --all_windows output/results.csv --negatives output/negatives.csv --balanced output/balanced.csv --plot_dir output/plots/
```

**Parameters:**
- `--positives`: CSV with positive windows (miRNA-containing)
- `--all_windows`: CSV with all genome-wide windows
- `--negatives`: Output CSV for sampled negatives
- `--balanced`: Output CSV with combined positives + negatives
- `--plot_dir`: Directory for comparison plots
- `--match_strand`: Match strand distribution (default: on)
- `--no_match_strand`: Disable strand matching
- `--match_chr`: Match chromosome distribution (default: on)
- `--no_match_chr`: Disable chromosome matching
- `--seed`: Random seed (default: 42)

**Matching features (22 total):**
- MFE (1)
- Dinucleotide frequencies (16): AA, AC, AG, AU, CA, CC, CG, CU, GA, GC, GG, GU, UA, UC, UG, UU
- Structure features (4): stem_length, loop_size, bulge_count, paired_fraction
- Sequence complexity (1): Shannon entropy (0=low complexity, 2=high complexity)

**Bias controls:**
- **Compositional matching**: Z-score normalized nearest-neighbor on all 22 features
- **Strand matching** (`--match_strand`): Sample negatives with same strand distribution as positives
- **Chromosome matching** (`--match_chr`): Sample negatives with same chromosome distribution as positives

**Outputs:**
- `negatives.csv`: Sampled negative windows
- `balanced_dataset.csv`: Combined dataset with 'label' column (positive/negative)
- `plots/mfe_comparison.png`: KDE plot comparing MFE distributions
- `plots/dinucleotide_comparison.png`: Bar chart comparing mean dinucleotide frequencies
- `plots/dinucleotide_kde_grid.png`: 4x4 grid of KDE plots for each dinucleotide
- `plots/structure_comparison.png`: 2x3 grid of structure feature distributions
- `plots/strand_comparison.png`: Bar chart of strand distribution (if --match_strand)
- `plots/chr_comparison.png`: Bar chart of chromosome distribution (if --match_chr)

## Train Random Forest Classifier

Train random forest model on balanced dataset (90-10 train-test split).

```bash
python scripts/train_rf.py \
  --input folding_output/balanced_dataset.csv \
  --model folding_output/rf_model.pkl \
  --report folding_output/rf_report.txt
```

**Parameters:**
- `--input`: Balanced dataset CSV
- `--model`: Output pickle file for trained model
- `--report`: Output text report
- `--test_size`: Test split ratio (default: 0.1)
- `--n_trees`: Number of trees (default: 100)
- `--random_state`: Random seed (default: 42)

**Features (16 total):**
- Dinucleotide frequencies: AA, AC, AG, AU, CA, CC, CG, CU, GA, GC, GG, GU, UA, UC, UG, UU

**Outputs:**
- `rf_model.pkl`: Trained random forest model
- `rf_report.txt`: Performance metrics (accuracy, precision, recall, f1, auroc, auprc), confusion matrix, feature importances
