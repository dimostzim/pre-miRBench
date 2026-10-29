#!/bin/bash
# pre-miRBench model v3 on node 5 (24 Oct 2026): score the agent's original iteration-19 model on the four tests;
# retrain it with pre-miRBench_model_v3/train.py (seeds 42 73 17, GPU); evaluate the retrained model on the four
# tests and the validation split (CPU); score the four genomes with it (CPU chunks) and compute their metrics.
# usage: RUN_DIR=<dir> premirbench_model_v3_queue.sh <workspace> <repo> <parallel chunks> <metrics python>
set -euo pipefail
W=$1; REPO=$2; PARALLEL=$3; MPY=$4
PY=/SCRATCH/dtzim01/conda-envs/premirbench_model_v2/bin/python
CODE=$REPO/pre-miRBench_model_v3
DATA=$W/agentomics/datasets/premirbench_v3
IDMAP=$W/agentomics/private/premirbench_v3_id_map.tsv
D=$W/datasets/mirgenedb_71/dataset.csv
GS=$W/genome_scale
ORIGINAL=$W/agentomics/original_v3/best_iteration_snapshot/model_training/training_artifacts
TOOL=premirbench_model_v3; RUN=v3_seed42
units=${RUN_DIR:?}/units.tsv
: > "$units"
export TMPDIR=$W/tmp_premirbench_model XDG_CACHE_HOME=$W/tmp_premirbench_model/cache PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$W/models/$TOOL" "$W/models/logs" "$W/genome_eval" "$W/genome_metrics"

echo "$(date -u +%FT%TZ) package and evaluate the original iteration-19 model"
(cd "$CODE" && "$PY" package_original.py --source "$ORIGINAL" --artifacts-dir "$W/models/$TOOL/original")
"$PY" -u "$CODE/evaluate.py" --dataset-dir "$DATA" --artifacts-dir "$W/models/$TOOL/original" \
  --output-dir "$W/evaluation/${TOOL}_original" --device cpu > "$W/evaluation/${TOOL}_original.log" 2>&1
printf "original\tok\t%s\n" "$W/evaluation/${TOOL}_original/metrics.csv" >> "$units"
echo "PROGRESS 1/7 original model evaluated"

out=$W/models/$TOOL/$RUN; ev=$W/evaluation/${TOOL}_$RUN; log=$W/models/logs/train_${TOOL}_$RUN.log
echo "$(date -u +%FT%TZ) train $TOOL $RUN"
if ! "$PY" -u "$CODE/train.py" --train-data "$DATA/train" --validation-data "$DATA/validation" --artifacts-dir "$out" \
     --device cuda --seeds 42 73 17 > "$log" 2>&1 || [ ! -f "$out/training_history.json" ]; then
  printf "train\tfailed\tsee %s\n" "$log" >> "$units"; echo "training failed, see $log"; exit 1
fi
printf "train\tok\t%s\n" "$out/training_history.json" >> "$units"
echo "PROGRESS 2/7 trained $TOOL $RUN"

echo "$(date -u +%FT%TZ) evaluate $TOOL $RUN"
"$PY" -u "$CODE/evaluate.py" --dataset-dir "$DATA" --artifacts-dir "$out" --output-dir "$ev" --device cpu > "$ev.log" 2>&1
"$PY" -u "$REPO/pipeline/premirbench_model_predictions.py" --dataset-dir "$DATA" --id-map "$IDMAP" --benchmark-dataset "$D" \
  --artifacts-dir "$out" --evaluation-dir "$ev" --model-dir "$CODE" --tool "$TOOL" > "$ev.predictions.log" 2>&1
printf "evaluation\tok\t%s\n" "$ev/predictions.csv" >> "$units"
echo "PROGRESS 3/7 evaluated $TOOL $RUN"

n=3
for sp in dme dan dya gga; do
  g=$W/genome_eval/${TOOL}_$sp; metrics=$W/genome_metrics/${TOOL}_$sp.csv
  echo "$(date -u +%FT%TZ) score $TOOL $sp"
  if [ ! -s "$g/predictions.csv" ]; then
    "$PY" -u "$REPO/pipeline/genome_scale/run_premirbench_model_genome.py" --repo "$REPO" --genome-dataset-dir "$GS/$sp" \
      --species "$sp" --artifacts-dir "$out" --python "$PY" --output-dir "$g" --parallel "$PARALLEL" --cores-per-chunk 4 \
      --model-dir "$CODE" --tool "$TOOL" > "$g.log" 2>&1 || true
  fi
  n=$((n + 1))
  if [ ! -s "$g/predictions.csv" ]; then
    printf "%s\tfailed\tno predictions.csv, see %s.log\n" "$sp" "$g" >> "$units"; echo "PROGRESS $n/7 $sp FAILED"; continue
  fi
  "$MPY" "$REPO/pipeline/genome_scale/genome_scale_metrics.py" --candidates "$GS/$sp/candidates.csv" --loci "$GS/$sp/loci.csv" \
    --genome-predictions "$g/predictions.csv" --validation-predictions "$ev/predictions.csv" --training-families "$D" \
    --genome-bases "$(grep folded_bases "$GS/candidates/$sp.stats.csv" | cut -d, -f2)" --output "$metrics" > "${metrics%.csv}.log" 2>&1 || true
  if [ -s "$metrics" ]; then
    printf "%s\tok\t%s\n" "$sp" "$metrics" >> "$units"
  else
    printf "%s\tfailed\tmetrics missing, see %s\n" "$sp" "${metrics%.csv}.log" >> "$units"
  fi
  echo "PROGRESS $n/7 $TOOL $sp"
done
echo "$(date -u +%FT%TZ) queue done"
