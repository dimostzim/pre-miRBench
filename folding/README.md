# RNA Folding Pipeline

Runs RNAfold on sliding windows across sequences and generates MFE distribution plot.

## Requirements

- ViennaRNA (`RNAfold` on PATH)
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
  --output results/ \
  --window 100 \
  --step 5 \
  --cpus 16
```

**Parameters:**
- `--input`: Input FASTA file
- `--output`: Output directory
- `--window`: Window size (default: 100)
- `--step`: Step size (default: 5)
- `--cpus`: Number of CPUs for parallel folding (optional)
- `--dna`: Keep DNA bases (default: converts T→U)

**Outputs:**
- `windows.fa`: Sliding windows FASTA
- `windows.fold`: RNAfold output (sequence + structure + MFE)
- `results.csv`: Window data (window_id, sequence, structure, mfe)
- `mfe_kde.png`: KDE plot with mean and Q25/Q75 percentile lines
