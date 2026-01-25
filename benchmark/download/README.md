# Download Data Scripts

## C. elegans Data

```bash
./download_celegans.sh [output_dir]
```

Downloads:
- `ce11.fa` - C. elegans genome
- `ce11.masked.fa` - Repeat-masked genome
- `ce11.gtf` - Gene annotation
- `cel-precursors-no-v2.bed` - miRNA precursor coordinates

## Human Data

```bash
./download_human.sh [chromosome] [output_dir]
```

Default: `chr14`

Downloads:
- `{chr}.fa` - Human chromosome sequence
- `{chr}.masked.fa` - Repeat-masked chromosome
- `hg38.gtf` - GTF annotation (filtered to chromosome)
- `hsa-precursors-no-v2.bed` - miRNA precursor coordinates (filtered to chromosome)
