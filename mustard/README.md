# MuStARD

## Setup
Installs MuStARD dependencies (Python 2.7, Perl modules, ViennaRNA, bedtools, R/ggplot2), clones source + pretrained models, and downloads chr14 genome + phyloP test data.

Requires conda: https://docs.anaconda.com/miniconda/install/

```bash
./setup.sh
```


## Usage
- Activate the environment:
```bash
conda activate mustard
```

Test on provided chr14 test data with the best miRNA model (MuStARD-mirSFC-U):
```bash
./run_inference.sh --input mustard_paper/test_loci/miRNA/hsa/hsa.hairpin.slop5k.bothStrands.chr14.bed --genome data/chr14.fa --consDir data --output results --chromList chr14 --model MuStARD-mirSFC-U
```

Parameters (MuStARD.pl predict):
- `--input` (required): input BED file.
- `--genome` (required): reference FASTA matching the BED coordinates.
- `--output` (required): output directory.
- `--model` (required): model name (e.g., MuStARD-mirSFC-U, MuStARD-mirSF).
- `--inputMode` (auto-set): inferred from the model (sequence / RNAfold / conservation combos).
- `--consDir` (required for models with C): directory with PhyloP conservation files.
- `--chromList` (optional): comma-separated chromosomes to restrict processing (e.g., chr14).
- `--threads` (optional, default 10): number of threads.
- `--winSize` (optional, default 100): sliding window size.
- `--step` (optional, default 5): step size.
- `--staticPredFlag` (optional, default 0): 0 = sliding-window scanning (winSize/step). 1 = score each input interval as-is (no tiling).
- `--modelType` (optional, default CNN): model type.
- `--classNum` (optional, default 2): number of classes.


## Models

| Model | Inputs |
|-------|--------|
| MuStARD-mirS | S |
| MuStARD-mirF | F |
| MuStARD-mirSF | S+F |
| MuStARD-mirSC | S+C |
| MuStARD-mirFC | F+C |
| MuStARD-mirSFC | S+F+C |
| MuStARD-mirSFC-U | S+F+C |

S=Sequence, F=RNAfold, C=Conservation

## Data
Full genome:
- Genome: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
- PhyloP: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/
