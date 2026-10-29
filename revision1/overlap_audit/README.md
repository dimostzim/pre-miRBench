# Revision 1 overlap audit (19 October 2026)

Read-only audit of the submitted pre-miRBench benchmark. Nothing in the dataset,
models, predictions or Zenodo release was changed.

Inputs:

- `datasets/mirgenedb_71-build/dataset.csv` (SHA-256 `98e369e2...`, matches run_metadata.json)
- `datasets/mirgenedb_71-build/genome.fa.fai` and `split_summary.csv`
- `zenodo_release_2026-09-17/paper_materials/code/repository/benchmarking_results/predictions/{published_tools,premirbench_model}.csv`

Environment: conda env `premirbench-review` (pandas, scikit-learn, blast, mmseqs2).
Run every script with `python -I <script> --help` for its arguments.

| Script | Question | Main outputs in `results/` |
|---|---|---|
| `overlap_audit.py` | Genomic window overlap and exact reverse complements between partitions; negative mining score by partition; chromosome position of negatives; family sharing; AP after removing overlapping test records versus random removal of the same number | `records_overlapping_train.tsv`, `genomic_overlap_pairs_summary.tsv`, `exact_sequence_matches_summary.tsv`, `negative_difficulty_by_split.tsv`, `chromosome_position_summary.tsv`, `family_overlap_between_splits.tsv`, `ap_after_overlap_exclusion.tsv` |
| `homology_audit.py` | blastn similarity of test precursors and test windows to training records | `precursor_similarity_to_train_summary.tsv`, `heldout_species_best_hit_species.tsv`, `window_similarity_to_train_summary.tsv` |
| `shortcut_baselines.py` | AP of simple baselines fitted on the training partition (MFE, hairpin features, mining features, k-mer composition, positional paired profile); `--predictions-dir` also writes per-record validation and test scores (v2 run: `results_v2/baseline_predictions/<baseline>/predictions.csv`) | `shortcut_baselines_ap.tsv` |
| `split_confounds.py` | AP when one tests positives are scored against another tests negatives (species-matched); bootstrap CIs resampling overlapping windows together | `negative_swap_ap.tsv`, `cluster_bootstrap_ap.tsv` |
| `verify_manuscript_numbers.py` | Recompute Tables 1-3 and in-text numbers from released files | `manuscript_number_check.txt` |

The review that interprets these results is in the Mac paper workspace:
`pre_mirbench_paper/manuscript/revision1_review_2026-10-19/review.md`.
