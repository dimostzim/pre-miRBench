# MirGeneDB 71-species exact panel

Final pre-miRBench species panel selected on 2026-07-03.

Definition: 71 usable exact species total, composed of 69 strict full-exact species plus 2 supplemented-contig exact species (`mml`, `pab`). For every included species, MirGeneDB precursor BED coordinates are usable against the selected genome FASTA, and extracted precursor sequences match MirGeneDB precursor sequences after T/U normalization.

The downloader uses `species_panel.tsv` as the source of truth. It downloads all MirGeneDB `_pre` rows, including v2/v3 precursor variants. `mml` and `pab` use the listed main genome FASTA plus the small FASTA/BED addenda under `supplements/`.

Default raw download target:

```bash
bash pipeline/download_data.sh
```

This writes `data/raw/mirgenedb_71/panel.tsv`.

## Dataset snapshot: 2026-07-04

The first 71-species leakage-controlled dataset was built on 2026-07-04 UTC
from this panel. The scratch copy is not committed to git, but the run metadata
and exact repo diff used for the run are saved beside the dataset:

- dataset: `/SCRATCH/dtzim01/pre-miRBench/datasets/mirgenedb_71`
- raw downloads: `/SCRATCH/dtzim01/pre-miRBench/raw/mirgenedb_71`
- work directory: `/SCRATCH/dtzim01/pre-miRBench/work/build_mirgenedb_71`
- metadata: `/SCRATCH/dtzim01/pre-miRBench/datasets/mirgenedb_71/run_metadata.json`
- repo diff at build time: `/SCRATCH/dtzim01/pre-miRBench/datasets/mirgenedb_71/repo_diff.patch`

Raw download manifest:

| Metric | Value |
| --- | ---: |
| Species | 71 |
| Auto species | 71 |
| Failed species | 0 |
| MirGeneDB precursor BED rows | 12,053 |
| Matched BED rows after chromosome normalization | 12,053 |
| Dropped BED rows | 0 |

Final dataset:

| Metric | Value |
| --- | ---: |
| Records | 77,616 |
| Unique 100 nt leakage keys | 77,616 |
| Excluded duplicate 100 nt positives | 1,259 |
| Negative:positive ratio | 10:1 in every split |
| Combined genome size | 92 GB |

Split counts:

| Split | Positives | Negatives |
| --- | ---: | ---: |
| `train` | 4,765 | 47,650 |
| `valid` | 631 | 6,310 |
| `test_known_species_known_family` | 707 | 7,070 |
| `test_known_species_heldout_family` | 677 | 6,770 |
| `test_heldout_species_known_family` | 207 | 2,070 |
| `test_heldout_species_heldout_family` | 69 | 690 |

The canonical leakage key is the exact prepared 100 nt sequence. No final row
shares that key with any other final row, including train/validation, train/test,
test/test, within-split duplicates, or positive/negative conflicts.

Positive de-duplication happened after positive split assignment. When multiple
positive rows had the same exact 100 nt sequence, the retained row was chosen by
split priority: `test_heldout_species_heldout_family`,
`test_heldout_species_known_family`, `test_known_species_heldout_family`,
`test_known_species_known_family`, `valid`, then `train`. This preserves the
strictest evaluation copy when a duplicate sequence appears in more than one
split. The 2026-07-04 build excluded 1,259 duplicate positive rows by this rule.

The build stdout was not captured to a persistent log file. Persistent run
artifacts are the raw `panel.tsv`, per-species normalization reports,
per-species BED/genome validation reports, per-species build stats, final split
summaries, `leakage_report.csv`, and the metadata JSON listed above.
