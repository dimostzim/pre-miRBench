# pre-miRBench revision 1, v3 benchmark: final results

Final outputs of the v3 benchmark rebuild, gathered on biogemt-6 from nodes 4, 5 and 6 (24 Oct 2026) and copied to
biogemt-4 `/SCRATCH/dtzim01/premirbench/v3/results`. Every gathered file was copied with its md5 checked against the
source; `manifest.tsv` gives the origin (node and path) and md5 of each. The genome curves of the methods scored on
node 6 (MuStARD, deepMiRGene, DeepMir on chicken, retrained MirMachine, ncOrtho) were written into this folder by
`scripts/genome_curves.sh` on node 6 and are not listed in `manifest.tsv`.

The node 5 and 6 workspaces (`/SCRATCH/dtzim01/premirbench-v3/`) were deleted on 26 Oct 2026. Their run folders are in
node 4 `v3/node_runs/node5/runs/` and `v3/node_runs/node6/runs/`; `v3/archive_manifest.tsv` lists every file copied from them.

## Layout

| Path | Contents |
|---|---|
| `benchmark/predictions/<model>/predictions.csv` | Per-record scores on the validation partition and the four tests, one folder per model (16): the six published predictors, retrained once with seed 1 (`<tool>_v3_rep1`), the pre-miRBench model v3 (`premirbench_model_v3_original`: the networks trained by the Agentomics run), the pre-miRBench model v2 recipe retrained on v3 with seed 42 (`premirbench_model_v2_v3_seed42`; not reported, so left out of `tables/`), retrained MirMachine with cutoffs fitted on the training genomes (`mirmachine_retrained_genome_cutoffs`; the 204 families without a fitting hit keep their window-fitted cutoff) and seven shortcut baselines (`baseline_*`) |
| `benchmark/summary/` | `summarize_benchmark.py` output for the 16 models (24 Oct): `replicate_metrics.csv`, `summary.csv` (AP and the 95% bootstrap CI over records; one model per method, so no SD), `paired_differences.csv`; run folder `runs/20261024-1238-benchmark-summary` on node 6 (copied to `../node_runs/node6/runs/`) |
| `benchmark/validation_ap.csv` | Validation-partition AP and AUROC of every model (`scripts/validation_ap.py`, the AP and AUROC functions of `summarize_benchmark.py`) |
| `genome/metrics/<tool>_<genome>.csv` | `genome_scale_metrics.py` output for the eight predictors (seed 1; the agent's networks for the pre-miRBench model v3, copied from `genome_metrics/premirbench_model_v3_original_<genome>.csv`; seed 42 for the v2 recipe), thresholds `valid_max_f1` and `valid_recall_90` |
| `genome/comparators/` | `compare_annotation_calls.py` summaries: retrained MirMachine with genome-fitted cutoffs (four genomes) and ncOrtho with training-family references (chicken) |
| `genome/pr_curves/<method>_<genome>.csv` | Genome precision-recall curve points (`scripts/genome_pr_points.py`): the points where recall rises, which give the AP exactly |
| `genome/family_ap/<method>_<genome>.csv` | Genome AP separately for loci of training families and of held-out families (`scripts/genome_family_ap.py`; same candidates and labels as `genome_scale_metrics.py`, the other group's positives left out) |
| `genome/discovery_curves/<method>_<genome>.csv` | MirGeneDB loci found among the top-ranked candidates, for all loci and for each family group (`scripts/genome_discovery_points.py`) |
| `tables/` | Co-author tables: `benchmark_tests_all_metrics.tsv` (every summary metric, one row per method and test), `genome_wide_all_metrics.tsv` (every genome metric including AP by family group, one row per method, genome and calling rule), `benchmark_tests.csv` and `genome_wide.csv` (figure inputs), the formatted workbook `premirbench_v3_results.xlsx`, and `all_results.tsv` with every number in long format; the result tables `table_benchmark_v3` (AP with 95% interval on the four tests), `table_genome_v3` (genome AP), `table_genome_family_v3` (genome AP and recall of unmasked loci, for training and for held-out families) and `table_genome_calls_v3` (calls per Mb and precision of the calls); in each column the best predictor is bold; MirMachine and ncOrtho, below a rule, are a reference block and are not bold (26 Oct); the titled copy of `table_genome_calls_v3` has a note with their precision per hit; each as a LaTeX tabular for the manuscript (`.tex`), a standalone document and its PDF, SVG and 300-dpi PNG, and the same with a title above the table (`*_titled.*`, for the co-author copies; the manuscript takes the title from the caption). Rebuilt on 25 Oct without the v2 model recipe; the 24 Oct tables, which include it, are in `tables_20261024_with_v2_recipe/` |
| `scripts/` | `gather_v3.sh` (run on the Mac; nodes 5 and 6 cannot SSH to each other), `benchmark_summary.sh`, `validation_ap.py`, `genome_curves.sh` with its job lists (`jobs_node5.txt`, `jobs_node6_a.txt`, `jobs_node6_b.txt`), `genome_pr_points.py`, `genome_family_ap.py`, `genome_discovery_points.py`, `make_tables_v3.py` (the 24 Oct version as `make_tables_v3.20261024.py`), `make_latex_tables_v3.py`, and `consolidate_v3.sh` (run on the Mac; copies the v3 outputs of nodes 5 and 6 to node 4) |

Everything else needed for a Zenodo version (models, logs, configs, evaluation folders, genome scores, MirMachine and
ncOrtho work, run folders and node scripts) is in the v3 tree around this folder on node 4, copied from nodes 5 and 6
by `scripts/consolidate_v3.sh`; `../archive_manifest.tsv` lists each copied file with its source and md5.

## How the files were made

- Benchmark summary (node 6, `premirbench-review` conda environment): `bash scripts/benchmark_summary.sh`, which runs
  `validation_ap.py` and then
  `summarize_benchmark.py --predictions results/benchmark/predictions/*/predictions.csv --output-dir results/benchmark/summary`
  (10,000 class-stratified bootstrap resamples, seed 42).
- Genome metrics: `genome_scale_metrics.py` on the node that scored each genome (node 5 `premirbench-metrics`, node 6
  `premirbench-review`), with the model's own validation predictions for the thresholds and the genome's
  `folded_bases` from `genome_scale/candidates/<genome>.stats.csv`.
- Genome curves: `genome_curves.sh` on node 5 (`jobs_node5.txt`) and node 6 (`jobs_node6_a.txt`, `jobs_node6_b.txt`),
  run folders `runs/20261024-1220-genome-curves-node5` (node 5), `runs/20261024-1223-genome-curves-node6-a` and
  `runs/20261024-1448-genome-curves-node6-b` (node 6).
- Tables: `python3 scripts/make_tables_v3.py --results-dir . --output-dir tables --xlsx tables/premirbench_v3_results.xlsx`
  (needs openpyxl; run on the Mac on a copy of the inputs, outputs copied back), then `python3 scripts/make_latex_tables_v3.py` next to
  the table CSVs and a `comparator_summaries/` copy of `genome/comparators/*.summary.csv` (needs pdflatex, pdftocairo and pdftoppm).

## Not included, by decision

- The pre-miRBench model v2 recipe retrained on v3: its predictions, summary rows and genome metrics are in this folder,
  but the tables leave it out; only the v3 model is reported (user decision, 25 Oct).

- The pre-miRBench model v3 retrained by us (node 5, seed 42; node 4 A10 retrain): the agent's trained networks are
  reported (user decision, 24 Oct). The retrained models and their evaluations are in `../models/premirbench_model_v3/`
  and `../evaluation/`.
- Off-the-shelf MirMachine: its models include the held-out species' precursors and the held-out families (user
  decision, 19 Oct).
- miRBase labels: MirGeneDB loci only (user decision, 19 Oct).

`../archive_manifest.tsv` records `tables/`, `README.md` and `scripts/make_tables_v3.py` as copied on 24 Oct; those
files were replaced on 25 Oct (the 24 Oct tables and script are kept under the names above).
