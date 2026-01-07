# MirGeneDB Human miRNA Orthologue Scraper

Automatically scrapes MirGeneDB to create FASTA files containing human miRNAs grouped with their cross-species orthologues.

## Quick Start

```bash
# Setup environment
conda env create -f environment.yml
conda activate mirna-orthologues

# Run
python scrape_orthologues.py --output_folder results --workers 16
```

## What It Does

1. Downloads all primary and precursor miRNA sequences from MirGeneDB
2. Scrapes the list of all human (Hsa) miRNAs
3. For each human miRNA:
   - Fetches its orthologue list from MirGeneDB
   - Groups sequences into FASTA files
4. Creates one FASTA per human miRNA containing all orthologues

## Output Structure

```
results/
├── downloads/
│   ├── all_primary.fa          # Downloaded bulk FASTA (all species)
│   └── all_precursor.fa
├── primary/
│   ├── Hsa-Mir-1.fasta         # Human Mir-1 + mouse/fish/chicken orthologues
│   ├── Hsa-Mir-10-P1b.fasta    # Human Mir-10-P1b + orthologues
│   ├── Hsa-Let-7-P1a.fasta     # Human Let-7-P1a + orthologues
│   └── ... (~500 files)
└── precursor/
    ├── Hsa-Mir-1.fasta
    ├── Hsa-Mir-10-P1b.fasta
    └── ... (~500 files)
```

### Example File Content

**`results/primary/Hsa-Mir-1.fasta`:**
```fasta
>Hsa-Mir-1_pri
ACGTACGTACGTACGT...
>Mmu-Mir-1_pri
ACGTACGTACGTACGT...
>Dre-Mir-1_pri
ACGTACGTACGTACGT...
>Gga-Mir-1_pri
ACGTACGTACGTACGT...
```

Each file contains the human miRNA followed by all confirmed orthologues found in the database.

## Command-Line Options

```
--output_folder PATH    Output directory (required)
--workers N             Number of parallel workers (default: 4)
--skip_download         Skip downloading FASTA files if already cached
```