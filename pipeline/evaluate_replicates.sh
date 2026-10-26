#!/bin/bash
# Score validation and the four test partitions with each trained replicate of the given tools.
# Inference runs without a GPU (--cpu-only, at most 8 cores per container) so it never competes with training on the same node.
# usage: [RUN_PREFIX=v3] evaluate_replicates.sh <repo> <dataset_dir> <training_root> <output_root> <tool>[,<tool>...] [replicates, default "1 2 3"]
set -euo pipefail
REPO=$1; DATASET=$2; TRAINING=$3; OUT=$4; TOOLS=$5; REPLICATES=${6:-1 2 3}
RUN_PREFIX=${RUN_PREFIX:-v2}  # run names <prefix>_rep<N>
SPLITS=valid,test_known_species_known_family,test_known_species_heldout_family,test_heldout_species_known_family,test_heldout_species_heldout_family
for tool in ${TOOLS//,/ }; do
  for rep in $REPLICATES; do
    run=${RUN_PREFIX}_rep$rep
    [ -f "$TRAINING/$tool/$run/inference_config.yaml" ] || { echo "missing $tool $run"; continue; }
    [ -s "$OUT/${tool}_${run}/predictions.csv" ] && { echo "skip $tool $run"; continue; }
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    # Output of an interrupted evaluation is moved aside, never reused.
    [ -e "$OUT/${tool}_${run}" ] && mv "$OUT/${tool}_${run}" "$OUT/${tool}_${run}_INCOMPLETE_${stamp}"
    [ -e "$OUT/${tool}_${run}.log" ] && mv "$OUT/${tool}_${run}.log" "$OUT/${tool}_${run}.INCOMPLETE_${stamp}.log"
    echo "$(date -u +%FT%TZ) evaluate $tool $run"
    python3 -u "$REPO/pipeline/evaluate.py" --dataset-dir "$DATASET" --training-root "$TRAINING" --run-name "$run" \
      --tools "$tool" --splits "$SPLITS" --output-dir "$OUT/${tool}_${run}" --cpu-only --cpu-threads 8 > "$OUT/${tool}_${run}.log" 2>&1 \
      || echo "FAILED $tool $run"
  done
done
