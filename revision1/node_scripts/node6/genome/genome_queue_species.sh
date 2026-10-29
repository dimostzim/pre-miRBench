#!/bin/bash
# v3 genome-scale scoring of one trained model, as genome_queue.sh, with two additions (23 Oct): SPECIES selects the
# genomes (default all four), so one model's genomes can be scored by two queues or on two nodes; SCORE_FIRST=1 starts
# scoring once the training run's units.tsv has "train<TAB>ok" and waits for the evaluation only before the metrics.
# usage: RUN_DIR=<dir> [SPECIES="dme dan dya gga"] [SCORE_FIRST=1] [FIRST_CORE=<n>] genome_queue_species.sh <workspace> <repo> <tool or premirbench_model_v2> <parallel chunks> <metrics python> <training run dir>
set -euo pipefail
W=$1; REPO=$2; TOOL=$3; PARALLEL=$4; MPY=$5; TRAIN_RUN=$6
SPECIES=${SPECIES:-dme dan dya gga}
GS=$W/genome_scale
D=$W/datasets/mirgenedb_71/dataset.csv
if [ "$TOOL" = premirbench_model_v2 ]; then RUN=v3_seed42; else RUN=v3_rep1; fi
EV=$W/evaluation/${TOOL}_${RUN}/predictions.csv
units=${RUN_DIR:?}/units.tsv
: > "$units"
total=$(echo $SPECIES | wc -w)
stop_if_failed() {
  if grep -qP "^($TOOL|train|evaluation)\tfailed" "$TRAIN_RUN/units.tsv" 2>/dev/null; then
    echo "$TOOL failed in $TRAIN_RUN/units.tsv; stopping"; exit 1
  fi
}
wait_for_evaluation() {
  echo "$(date -u +%FT%TZ) waiting for $EV"
  until [ -s "$EV" ]; do stop_if_failed; sleep 300; done
  # The evaluation file is complete once the process writing it has ended.
  while pgrep -f "evaluate.py .*--tools $TOOL --splits|premirbench_model_predictions.py .*premirbench_model_v2/$RUN" > /dev/null; do sleep 120; done
}
echo "$(date -u +%FT%TZ) waiting for $GS/READY"
until [ -e "$GS/READY" ]; do sleep 300; done
if [ "${SCORE_FIRST:-0}" = 1 ]; then
  echo "$(date -u +%FT%TZ) waiting for train ok in $TRAIN_RUN/units.tsv"
  until grep -qP "^train\tok" "$TRAIN_RUN/units.tsv" 2>/dev/null; do stop_if_failed; sleep 300; done
else
  wait_for_evaluation
fi
mkdir -p "$W/genome_eval" "$W/genome_metrics"
for sp in $SPECIES; do
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
  echo "$(date -u +%FT%TZ) scored $TOOL $sp: $( [ -s "$out/predictions.csv" ] && echo ok || echo FAILED)"
done
[ "${SCORE_FIRST:-0}" = 1 ] && wait_for_evaluation
n=0
for sp in $SPECIES; do
  out=$W/genome_eval/${TOOL}_$sp
  if [ ! -s "$out/predictions.csv" ]; then
    printf "%s\tfailed\tno predictions.csv, see %s.log\n" "$sp" "$out" >> "$units"; n=$((n + 1)); echo "PROGRESS $n/$total $sp FAILED"; continue
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
  n=$((n + 1)); echo "PROGRESS $n/$total $TOOL $sp"
done
echo "$(date -u +%FT%TZ) genome queue done $TOOL $SPECIES"
