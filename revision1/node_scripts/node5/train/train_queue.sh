#!/bin/bash
# v3 training queue for one node: published tools one replicate each (seed 1, run name v3_rep1), one after another
# on the GPU; each trained model is evaluated on CPU in the background while the next one trains. With --pmb, the
# pre-miRBench model (seed 42) is trained last on the Agentomics export of the same dataset.
# A run counts as trained only with the inference config, the validation-AUPRC selection line and no traceback.
# usage: RUN_DIR=<run dir> train_queue.sh <workspace> <repo> <dataset_dir> <tool>[,<tool>...] [--pmb]
set -euo pipefail
W=$1; REPO=$2; D=$3; TOOLS=$4; PMB=${5:-}
export RUN_PREFIX=v3
units=${RUN_DIR:?}/units.tsv
: > "$units"
mkdir -p "$W/models/logs" "$W/evaluation"
total=$(echo "${TOOLS//,/ }" | wc -w); [ "$PMB" = --pmb ] && total=$((total + 1))
n=0
eval_pids=(); eval_tools=()
for tool in ${TOOLS//,/ }; do
  log=$W/models/logs/train_${tool}_v3_rep1.log
  echo "$(date -u +%FT%TZ) train $tool"
  if ! bash "$REPO/pipeline/train_replicates.sh" "$D" "$W/models" "$tool" 1; then
    printf "%s\tfailed\ttraining exited non-zero, see %s\n" "$tool" "$log" >> "$units"; n=$((n + 1)); echo "PROGRESS $n/$total $tool FAILED"; continue
  fi
  case $tool in mirdnn) selection="valid_auprc=" ;; mire2e) selection="Best epoch [0-9]+: AUPRC" ;; *) selection="val_auprc: " ;; esac
  if [ ! -f "$W/models/$tool/v3_rep1/inference_config.yaml" ] || grep -q "^Traceback (most recent call last)" "$log" || ! grep -aqE "$selection" "$log"; then
    printf "%s\tfailed\tno inference config, a traceback or no validation-AUPRC selection line in %s\n" "$tool" "$log" >> "$units"
    n=$((n + 1)); echo "PROGRESS $n/$total $tool FAILED"; continue
  fi
  echo "$(date -u +%FT%TZ) trained $tool; evaluating in the background"
  bash "$REPO/pipeline/evaluate_replicates.sh" "$REPO" "$D" "$W/models" "$W/evaluation" "$tool" 1 > "$W/evaluation/evaluate_${tool}_v3_rep1.queue.log" 2>&1 &
  eval_pids+=($!); eval_tools+=("$tool")
  n=$((n + 1)); echo "PROGRESS $n/$total trained $tool"
done

if [ "$PMB" = --pmb ]; then
  PY=/SCRATCH/dtzim01/conda-envs/premirbench_model_v2/bin/python
  CODE=$REPO/pre-miRBench_model_v2
  DATA=$W/agentomics/datasets/premirbench_v3
  IDMAP=$W/agentomics/private/premirbench_v3_id_map.tsv
  run=v3_seed42; out=$W/models/premirbench_model_v2/$run; ev=$W/evaluation/premirbench_model_v2_$run
  log=$W/models/logs/train_premirbench_model_v2_$run.log
  export TMPDIR=$W/tmp_premirbench_model XDG_CACHE_HOME=$W/tmp_premirbench_model/cache PYTHONDONTWRITEBYTECODE=1
  mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$(dirname "$IDMAP")" "$W/models/premirbench_model_v2"
  if [ ! -d "$DATA" ]; then
    python3 -u "$REPO/pipeline/export_agentomics_dataset.py" --dataset-csv "$D/dataset.csv" --output-dir "$DATA" \
      --dataset-name premirbench_v3 --id-map "$IDMAP"
  fi
  echo "$(date -u +%FT%TZ) train premirbench_model_v2 $run"
  if "$PY" -u "$CODE/train.py" --train-data "$DATA/train" --validation-data "$DATA/validation" --artifacts-dir "$out" \
       --device cuda --seed 42 > "$log" 2>&1 && [ -f "$out/training_history.json" ]; then
    n=$((n + 1)); echo "PROGRESS $n/$total trained premirbench_model_v2"
    echo "$(date -u +%FT%TZ) evaluate premirbench_model_v2 $run"
    "$PY" -u "$CODE/evaluate.py" --dataset-dir "$DATA" --artifacts-dir "$out" --output-dir "$ev" --device cpu > "$ev.log" 2>&1
    "$PY" -u "$REPO/pipeline/premirbench_model_predictions.py" --dataset-dir "$DATA" --id-map "$IDMAP" \
      --benchmark-dataset "$D/dataset.csv" --artifacts-dir "$out" --evaluation-dir "$ev" > "$ev.predictions.log" 2>&1
    if [ -s "$ev/predictions.csv" ]; then
      printf "premirbench_model_v2\tok\t%s\n" "$ev/predictions.csv" >> "$units"
    else
      printf "premirbench_model_v2\tfailed\tno predictions.csv, see %s.log\n" "$ev" >> "$units"
    fi
  else
    printf "premirbench_model_v2\tfailed\ttraining failed, see %s\n" "$log" >> "$units"
    n=$((n + 1)); echo "PROGRESS $n/$total premirbench_model_v2 FAILED"
  fi
fi

for i in "${!eval_pids[@]}"; do
  tool=${eval_tools[$i]}
  wait "${eval_pids[$i]}" || true
  if [ -s "$W/evaluation/${tool}_v3_rep1/predictions.csv" ]; then
    printf "%s\tok\t%s\n" "$tool" "$W/evaluation/${tool}_v3_rep1/predictions.csv" >> "$units"
  else
    printf "%s\tfailed\tevaluation wrote no predictions.csv, see %s\n" "$tool" "$W/evaluation/${tool}_v3_rep1.log" >> "$units"
  fi
done
echo "$(date -u +%FT%TZ) queue done"
