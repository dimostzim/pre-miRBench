#!/bin/bash
# Train each requested tool three times on one dataset (replicates rep1-rep3), with seed = replicate number for
# every tool (MuStARD's hard-coded seeds are replaced through its wrapper's --seed).
# Completed replicates (with an inference_config.yaml) are skipped. A replicate folder or log left by an
# interrupted run is renamed <name>_INCOMPLETE_<UTC time>, never reused or overwritten.
# DeepMir replicates share one hairpin-image cache (<output_root>/deepmir/_image_cache); images depend only on the
# record's sequence.
#
# usage: [RUN_PREFIX=v3] pipeline/train_replicates.sh <dataset_dir> <output_root> <tool>[,<tool>...] [replicates, default "1 2 3"]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="$1"
OUTPUT_ROOT="$2"
TOOLS="$3"
REPLICATES="${4:-1 2 3}"
RUN_PREFIX="${RUN_PREFIX:-v2}"  # run names <prefix>_rep<N>
mkdir -p "$OUTPUT_ROOT/configs" "$OUTPUT_ROOT/logs"

for tool in ${TOOLS//,/ }; do
  for rep in $REPLICATES; do
    run="${RUN_PREFIX}_rep${rep}"
    out="$OUTPUT_ROOT/$tool/$run"
    log="$OUTPUT_ROOT/logs/train_${tool}_${run}.log"
    if [ -f "$out/inference_config.yaml" ]; then
      echo "skip $tool $run (complete)"
      continue
    fi
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    if [ -e "$out" ]; then
      mv "$out" "${out}_INCOMPLETE_${stamp}"
      echo "moved partial $out -> ${out}_INCOMPLETE_${stamp}"
    fi
    if [ -e "$log" ]; then
      mv "$log" "${log%.log}.INCOMPLETE_${stamp}.log"
    fi
    config="$OUTPUT_ROOT/configs/${tool}.${run}.yaml"
    echo "seed: ${rep}" > "$config"
    if [ "$tool" = "deepmir" ]; then
      mkdir -p "$OUTPUT_ROOT/deepmir/_image_cache"
      echo "image_cache_dir: $OUTPUT_ROOT/deepmir/_image_cache" >> "$config"
    fi
    echo "$(date -u +%FT%TZ) start $tool $run"
    python3 -u "$REPO/pipeline/train.py" --tool "$tool" --run-name "$run" --dataset-dir "$DATASET" \
      --output-root "$OUTPUT_ROOT" --config "$config" > "$log" 2>&1
    echo "$(date -u +%FT%TZ) done $tool $run"
  done
done
