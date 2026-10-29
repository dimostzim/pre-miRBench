#!/bin/bash
# v3 genome-scale scoring of one trained model: the four held-out genomes on CPU (chunks), then genome_scale_metrics.py
# per genome with the model's validation threshold. Starts when the genome candidates are in place (<workspace>/genome_scale/READY)
# and the model's benchmark evaluation has written its predictions; stops if the training queue marked the model failed.
# usage: RUN_DIR=<dir> [FIRST_CORE=<n>] genome_queue.sh <workspace> <repo> <tool or premirbench_model_v2> <parallel chunks> <metrics python> <training run dir>
set -euo pipefail
W=$1; REPO=$2; TOOL=$3; PARALLEL=$4; MPY=$5; TRAIN_RUN=$6
GS=$W/genome_scale
D=$W/datasets/mirgenedb_71/dataset.csv
if [ "$TOOL" = premirbench_model_v2 ]; then RUN=v3_seed42; else RUN=v3_rep1; fi
EV=$W/evaluation/${TOOL}_${RUN}/predictions.csv
units=${RUN_DIR:?}/units.tsv
: > "$units"
echo "$(date -u +%FT%TZ) waiting for $GS/READY and $EV"
until [ -e "$GS/READY" ] && [ -s "$EV" ]; do
  if grep -qP "^$TOOL\tfailed" "$TRAIN_RUN/units.tsv" 2>/dev/null; then
    echo "$TOOL failed in $TRAIN_RUN/units.tsv; no genome scoring"; exit 1
  fi
  sleep 300
done
# The evaluation file is complete once the process writing it has ended.
while pgrep -f "evaluate.py .*--tools $TOOL --splits|premirbench_model_predictions.py .*premirbench_model_v2/$RUN" > /dev/null; do sleep 120; done
mkdir -p "$W/genome_eval" "$W/genome_metrics"
n=0
for sp in dme dan dya gga; do
  out=$W/genome_eval/${TOOL}_$sp
  if [ ! -s "$out/predictions.csv" ]; then
    echo "$(date -u +%FT%TZ) score $TOOL $sp"
    if [ "$TOOL" = premirbench_model_v2 ]; then
      PY=/SCRATCH/dtzim01/conda-envs/premirbench_model_v2/bin/python
      "$PY" -u "$REPO/pipeline/genome_scale/run_premirbench_model_genome.py" --repo "$REPO" --genome-dataset-dir "$GS/$sp" --species "$sp" \
        --artifacts-dir "$W/models/premirbench_model_v2/$RUN" --python "$PY" --output-dir "$out" \
        --parallel "$PARALLEL" --cores-per-chunk 4 --first-core "${FIRST_CORE:-0}" > "$out.log" 2>&1 || true
    else
      python3 -u "$REPO/pipeline/genome_scale/run_genome_inference.py" --repo "$REPO" --genome-dataset-dir "$GS/$sp" --species "$sp" \
        --tool "$TOOL" --training-root "$W/models" --run-name "$RUN" --output-dir "$out" --parallel "$PARALLEL" --cpu-only \
        > "$out.log" 2>&1 || true
    fi
  fi
  if [ ! -s "$out/predictions.csv" ]; then
    printf "%s\tfailed\tno predictions.csv, see %s.log\n" "$sp" "$out" >> "$units"; n=$((n + 1)); echo "PROGRESS $n/4 $sp FAILED"; continue
  fi
  metrics=$W/genome_metrics/${TOOL}_$sp.csv
  "$MPY" "$REPO/pipeline/genome_scale/genome_scale_metrics.py" --candidates "$GS/$sp/candidates.csv" --loci "$GS/$sp/loci.csv" \
    --genome-predictions "$out/predictions.csv" --validation-predictions "$EV" --training-families "$D" \
    --genome-bases "$(grep folded_bases "$GS/candidates/$sp.stats.csv" | cut -d, -f2)" --output "$metrics" > "${metrics%.csv}.log" 2>&1 || true
  if [ -s "$metrics" ]; then
    printf "%s\tok\t%s\n" "$sp" "$metrics" >> "$units"
  else
    printf "%s\tfailed\tmetrics missing, see %s.log\n" "$sp" "${metrics%.csv}" >> "$units"
  fi
  n=$((n + 1)); echo "PROGRESS $n/4 $TOOL $sp"
done
echo "$(date -u +%FT%TZ) genome queue done $TOOL"
