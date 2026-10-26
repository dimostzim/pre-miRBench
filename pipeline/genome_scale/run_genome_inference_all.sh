#!/bin/bash
# Genome-scale inference for one published predictor over the held-out genomes, one genome at a time.
# Scoring runs without a GPU (--cpu-only, 4 cores per chunk container) so it never competes with training. Finished genomes (predictions.csv present) are skipped; an already running
# run_genome_inference.py for the same tool and genome is waited for first, so this can be queued behind it while other genomes run separately.
# usage: run_genome_inference_all.sh <repo> <genome_root> <training_root> <output_root> <tool> <run_name> <parallel> [genomes, default "dme dan dya gga"]
set -euo pipefail
REPO=$1; GS=$2; TRAINING=$3; OUT=$4; TOOL=$5; RUN=$6; PARALLEL=$7; GENOMES="${8:-dme dan dya gga}"
for sp in $GENOMES; do
  while pgrep -f "run_genome_inference.py .*--species $sp --tool $TOOL " > /dev/null; do sleep 60; done
  [ -s "$OUT/${TOOL}_$sp/predictions.csv" ] && { echo "skip $TOOL $sp"; continue; }
  echo "$(date -u +%FT%TZ) start $TOOL $sp"
  python3 -u "$REPO/pipeline/genome_scale/run_genome_inference.py" --repo "$REPO" --genome-dataset-dir "$GS/$sp" --species "$sp" \
    --tool "$TOOL" --training-root "$TRAINING" --run-name "$RUN" --output-dir "$OUT/${TOOL}_$sp" --parallel "$PARALLEL" --cpu-only \
    > "$OUT/${TOOL}_$sp.log" 2>&1
  echo "$(date -u +%FT%TZ) done $TOOL $sp"
done
