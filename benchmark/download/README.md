# Download Data Scripts

## Common Species Data

```bash
./download_species.sh <species_code> [output_dir] [chromosome|all]
```

Supported species:

| Code | Species | UCSC build | MirGeneDB check |
|------|---------|------------|-----------------|
| `hsa` | human | `hg38` | available |
| `mmu` | mouse | `mm39` | available |
| `rno` | rat | `rn7` | available |
| `dme` | fruit fly | `dm6` | available |
| `dre` | zebrafish | `danRer11` | available |
| `cel` | C. elegans | `ce11` | available |

Examples:

```bash
./download_species.sh mmu data/train/raw/mmu all
./download_species.sh hsa data/train/raw/hsa_chr14 chr14
```

The script downloads:

- `<ucsc_build>.fa` - genome FASTA
- `<species_code>-precursors-no-v2.bed` - filtered MirGeneDB `_pre` entries

To check MirGeneDB BED availability/counts for species codes:

```bash
./check_mirgenedb_species.sh hsa,mmu,rno,dme,dre,cel
```

The generic species script does not download conservation tracks. MuStARD can
still train without conservation by using `inputMode: sequence,RNAfold` or
`inputMode: sequence`, unless species-specific conservation files are supplied.

## Diverse 20-Species Panel

```bash
./download_diverse20.sh [output_root]
```

Default output root: `data/train/raw/diverse20`.

The panel is intended to avoid overloading the benchmark with closely related
species. It includes broad vertebrate and invertebrate coverage.

The script automatically downloads genome FASTA, MirGeneDB BED, and runs
BED-vs-genome validation for:

```text
hsa, mmu, mdo, oan, gga, aca, xtr, dre, cmi, bfl, cin, dme, aga, cel, spu
```

It downloads MirGeneDB BEDs but marks manual genome sourcing as required for:

```text
loc, tca, nve, aqu, sro
```

because those species do not have exact UCSC genome matches in the UCSC genome
API used by this script.

The panel writes:

```text
<output_root>/panel.tsv
<output_root>/<species>/<ucsc_build>.fa
<output_root>/<species>/<species>-precursors-no-v2.bed
<output_root>/<species>/bed_genome_validation.txt
```

You can rerun validation manually:

```bash
python benchmark/train_data/validate_bed_genome.py \
  --bed data/train/raw/diverse20/hsa/hsa-precursors-no-v2.bed \
  --genome data/train/raw/diverse20/hsa/hg38.fa
```

## C. elegans Data

```bash
./download_celegans.sh [output_dir]
```

Downloads:
- `ce11.fa` - C. elegans genome (soft-masked: repeats in lowercase)
- `ce11.gtf` - Gene annotation
- `cel-precursors-no-v2.bed` - miRNA precursor coordinates

## Human Data

```bash
./download_human.sh [chromosome] [output_dir]
```

Default: `chr14`

Downloads:
- `{chr}.fa` - Human chromosome sequence (soft-masked: repeats in lowercase)
- `{chr}.wigFix.gz` - PhyloP basewise conservation track for MuStARD
- `hg38.gtf` - GTF annotation (filtered to chromosome)
- `hsa-precursors-no-v2.bed` - miRNA precursor coordinates (filtered to chromosome)
