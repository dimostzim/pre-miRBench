# MirGeneDB 71-species exact panel

Final pre-miRBench species panel selected on 2026-07-03.

Definition: 71 usable exact species total, composed of 69 strict full-exact species plus 2 supplemented-contig exact species (`mml`, `pab`). For every included species, MirGeneDB precursor BED coordinates are usable against the selected genome FASTA, and extracted precursor sequences match MirGeneDB precursor sequences after T/U normalization.

The downloader uses `species_panel.tsv` as the source of truth. It downloads all MirGeneDB `_pre` rows, including v2/v3 precursor variants. `mml` and `pab` use the listed main genome FASTA plus the small FASTA/BED addenda under `supplements/`.

Default raw download target:

```bash
bash pipeline/download_data.sh
```

This writes `data/raw/mirgenedb_71/panel.tsv`.
