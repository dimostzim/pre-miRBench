#!/bin/bash
# v3 benchmark summary (24 Oct 2026), as v2's summary_30_models.sh: validation AP of every model, then
# summarize_benchmark.py (10,000 class-stratified bootstrap resamples, seed 42) over results/benchmark/predictions.
set -euo pipefail
W=/SCRATCH/dtzim01/premirbench-v3; PY=/SCRATCH/dtzim01/conda-envs/premirbench-review/bin/python
cd "$W"
"$PY" results_scripts/validation_ap.py repo results
"$PY" repo/pipeline/summarize_benchmark.py --predictions results/benchmark/predictions/*/predictions.csv --output-dir results/benchmark/summary
echo "exit 0"
