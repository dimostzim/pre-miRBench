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
| `cfa` | dog | `canFam3` | available |
| `cpo` | guinea pig | `cavPor3` | available |
| `ocu` | rabbit | `oryCun2` | available |
| `eca` | horse | `equCab3` | available |
| `mdo` | opossum | `monDom5` | available |
| `bta` | cow | `bosTau9` | available |
| `gga` | chicken | `galGal6` | available |
| `tgu` | zebra finch | `taeGut2` | available |
| `aca` | anole lizard | `anoCar2` | available |
| `cpi` | painted turtle | `chrPic1` | available |
| `xtr` | X. tropicalis | `xenTro10` | available |
| `xla` | X. laevis | `xenLae2` | available |
| `lch` | coelacanth | `latCha1` | available |
| `cmi` | elephant shark | `calMil1` | available |
| `tni` | Tetraodon | `tetNig2` | available |
| `cin` | Ciona | `ci3` | available |
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

The script downloads genome FASTA and MirGeneDB BED files for all 20 panel
species, normalizes BED contig names to the downloaded genome FASTA headers,
and runs BED-vs-genome validation:

```text
hsa, mmu, cfa, cpo, ocu, eca, mdo, bta, gga, tgu,
aca, cpi, xtr, lch, dre, cmi, tni, cin, dme, cel
```

These species have MirGeneDB BEDs and UCSC genome FASTA downloads that validate
after BED contig-name normalization. The panel deliberately avoids stacking
closely related primates.

The panel writes:

```text
<output_root>/panel.tsv
<output_root>/<species>/<ucsc_build>.fa
<output_root>/<species>/<species>-precursors-no-v2.bed
<output_root>/<species>/<species>-precursors-no-v2.normalized.bed
<output_root>/<species>/bed_chrom_normalization.txt
<output_root>/<species>/bed_genome_validation.txt
```

`panel.tsv` points at the normalized BED. The normalization handles common
UCSC differences such as `1` -> `chr1` and `X` -> `chrX`. Contigs that still
cannot be matched are dropped and reported in `bed_chrom_normalization.txt`.
The manifest also records `bed_rows`, `matched_rows`, and `dropped_rows` for
each species.

You can rerun validation manually:

```bash
python benchmark/train_data/validate_bed_genome.py \
  --bed data/train/raw/diverse20/hsa/hsa-precursors-no-v2.normalized.bed \
  --genome data/train/raw/diverse20/hsa/hg38.fa
```

After download, build the prefixed multispecies training dataset:

```bash
python benchmark/train_data/build_multispecies_dataset.py \
  --panel data/train/raw/diverse20/panel.tsv \
  --output-dir data/train/diverse20 \
  --work-dir benchmark/train_data/work_diverse20 \
  --ratio 5 \
  --window 200 \
  --step 50 \
  --heldout-species dre,dme \
  --max-negative-windows-per-species 50000 \
  --cpus 8
```

The multispecies builder prefixes every contig as `<species>__<contig>` and
creates these canonical splits:

- `train`
- `valid` - one validation chromosome/scaffold per non-heldout species, when available
- `test_chrom` - one test chromosome/scaffold per non-heldout species, when available
- `test_species` - all rows from the held-out species

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
