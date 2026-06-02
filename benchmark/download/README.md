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
