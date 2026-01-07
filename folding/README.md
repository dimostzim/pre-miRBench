# RNA Folding Pipeline

Runs RNAfold on sliding windows across sequences and generates MFE distribution plot.

## Requirements

- ViennaRNA (`RNAfold` on PATH)
- bedtools
- Python packages: `numpy`, `matplotlib`

## Setup

```bash
conda env create -f environment.yml
conda activate folding
```

## Download C. elegans Genome

```bash
./download_c_elegans_genome.sh
```

Downloads UCSC `ce11` to `data/ce11.fa`.

## Usage

```bash
python run_folding.py \
  --input genome.fa \
  --output folding_output/ \
  --window 100 \
  --step 5 \
  --cpus 16
```

**Parameters:**
- `--input`: Input FASTA file
- `--output`: Output directory
- `--window`: Window size (default: 200)
- `--step`: Step size (default: 50)
- `--cpus`: Number of CPUs for parallel folding (default: None)
- `--dna`: Keep DNA bases (default: converts T→U)

**Outputs:**
- `windows.fa`: Sliding windows FASTA
- `windows.fold`: RNAfold output (sequence + structure + MFE)
- `results.csv`: Window data (window_id, sequence, structure, mfe)
- `mfe_kde.png`: KDE plot with mean and mean±SD lines

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

Sample negative windows matching MFE distribution of positives (1:1 ratio).

```bash
python scripts/sample_negatives.py \
  --positives folding_output/mirna_windows.csv \
  --all_windows folding_output/results.csv \
  --negatives folding_output/negatives.csv \
  --balanced folding_output/balanced_dataset.csv \
  --plot folding_output/mfe_comparison.png
```

**Parameters:**
- `--positives`: CSV with positive windows (miRNA-containing)
- `--all_windows`: CSV with all genome-wide windows
- `--negatives`: Output CSV for sampled negatives
- `--balanced`: Output CSV with combined positives + negatives
- `--plot`: MFE comparison plot
- `--bins`: Number of MFE bins for stratified sampling (default: 50)

**Outputs:**
- `negatives.csv`: Sampled negative windows with matched MFE distribution
- `balanced_dataset.csv`: Combined dataset with 'label' column (positive/negative)
- `mfe_comparison.png`: KDE plot comparing positive vs negative distributions
