# Revision 1: v3 workspace and analysis scripts

The pipeline code of the v3 benchmark is in `pipeline/` and the model packages in `pre-miRBench_model_v3/`. This folder
holds the scripts that ran around them on the biogemt nodes, as they were run. They use the absolute paths of the
workspaces (`/SCRATCH/dtzim01/premirbench/v3` on biogemt-4, `/SCRATCH/dtzim01/premirbench-v3` on biogemt-5 and -6).

| Folder | Contents | Origin |
|---|---|---|
| `workspace/` | Run launcher (`bin/run.sh`), dataset build (`build/run_build.sh`), genome-scale candidates (`genome_scale/run_candidates.sh`), window-centring checks (`validation/`), MirMachine 2 and ncOrtho runs (`comparators/`), the Agentomics run (`agentomics/run_agentomics_v3_run1.sh`) | biogemt-4 `v3/` |
| `node_scripts/` | Training, evaluation, genome-scoring and curve queues of biogemt-5 and -6 | biogemt-4 `v3/node_scripts/` (copied from the node workspaces, deleted on 26 Oct 2026) |
| `analysis/` | Gathering and copying of the results (`gather_v3.sh`, `consolidate_v3.sh`), benchmark summary, genome curves and AP by family group, result tables (`make_tables_v3.py`, `make_latex_tables_v3.py`); `figures/` holds the figure scripts; `results_README.md` describes the results folder they read and write | biogemt-4 `v3/results/scripts/` and `v3/results/README.md`; figures from the paper workspace |
| `overlap_audit/` | Overlap, homology and split audits, shortcut baselines, similarity of test positives to training positives | biogemt-4 `revision1_overlap_audit/` |
| `combination/` | pre-miRBench combined with MirMachine 2: the fixed rule (`combine_scores.py`, `run_combination.sh`) and the combiner fitted on the validation windows (`combiner.py`, `run_window_combiner.sh`), reported as a supplementary analysis; `run_genome_validation.sh`, `genome_validation_set.py`, `mirmachine_training_genome_gff.py` and `run_genome_combiner.sh` were for a genome-scale fit that was stopped after 2 of 15 genomes and is not reported | biogemt-4 `v3/combination_mirmachine_premirbench/` |
