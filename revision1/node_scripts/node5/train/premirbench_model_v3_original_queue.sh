#!/bin/bash
# pre-miRBench model v3 on node 5 (24 Oct 2026): the agent's original iteration-19 model (packaged by
# premirbench_model_v3_queue.sh as models/premirbench_model_v3/original and evaluated on the four tests):
# validation predictions and predictions.csv (CPU), genome scoring of the four genomes (CPU chunks) and metrics.
# usage: RUN_DIR=<dir> premirbench_model_v3_original_queue.sh <workspace> <repo> <parallel chunks> <metrics python>
set -euo pipefail
W=$1; REPO=$2; PARALLEL=$3; MPY=$4
PY=/SCRATCH/dtzim01/conda-envs/premirbench_model_v2/bin/python
CODE=$REPO/pre-miRBench_model_v3
DATA=$W/agentomics/datasets/premirbench_v3
IDMAP=$W/agentomics/private/premirbench_v3_id_map.tsv
D=$W/datasets/mirgenedb_71/dataset.csv
GS=$W/genome_scale
TOOL=premirbench_model_v3
out=$W/models/$TOOL/original; ev=$W/evaluation/${TOOL}_original
units=${RUN_DIR:?}/units.tsv
: > "$units"
export TMPDIR=$W/tmp_premirbench_model XDG_CACHE_HOME=$W/tmp_premirbench_model/cache PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$W/genome_eval" "$W/genome_metrics"
[ -s "$ev/metrics.csv" ] || { echo "$ev/metrics.csv missing"; exit 1; }

echo "$(date -u +%FT%TZ) predictions.csv for the original model"
"$PY" -u "$REPO/pipeline/premirbench_model_predictions.py" --dataset-dir "$DATA" --id-map "$IDMAP" --benchmark-dataset "$D" \
  --artifacts-dir "$out" --evaluation-dir "$ev" --model-dir "$CODE" --tool "$TOOL" > "$ev.predictions.log" 2>&1
printf "evaluation\tok\t%s\n" "$ev/predictions.csv" >> "$units"
echo "PROGRESS 1/5 predictions.csv"

n=1
for sp in dme dan dya gga; do
  g=$W/genome_eval/${TOOL}_original_$sp; metrics=$W/genome_metrics/${TOOL}_original_$sp.csv
  echo "$(date -u +%FT%TZ) score $TOOL original $sp"
  if [ ! -s "$g/predictions.csv" ]; then
    "$PY" -u "$REPO/pipeline/genome_scale/run_premirbench_model_genome.py" --repo "$REPO" --genome-dataset-dir "$GS/$sp" \
      --species "$sp" --artifacts-dir "$out" --python "$PY" --output-dir "$g" --parallel "$PARALLEL" --cores-per-chunk 4 \
      --model-dir "$CODE" --tool "$TOOL" > "$g.log" 2>&1 || true
  fi
  n=$((n + 1))
  if [ ! -s "$g/predictions.csv" ]; then
    printf "%s\tfailed\tno predictions.csv, see %s.log\n" "$sp" "$g" >> "$units"; echo "PROGRESS $n/5 $sp FAILED"; continue
  fi
  "$MPY" "$REPO/pipeline/genome_scale/genome_scale_metrics.py" --candidates "$GS/$sp/candidates.csv" --loci "$GS/$sp/loci.csv" \
    --genome-predictions "$g/predictions.csv" --validation-predictions "$ev/predictions.csv" --training-families "$D" \
    --genome-bases "$(grep folded_bases "$GS/candidates/$sp.stats.csv" | cut -d, -f2)" --output "$metrics" > "${metrics%.csv}.log" 2>&1 || true
  if [ -s "$metrics" ]; then
    printf "%s\tok\t%s\n" "$sp" "$metrics" >> "$units"
  else
    printf "%s\tfailed\tmetrics missing, see %s\n" "$sp" "${metrics%.csv}.log" >> "$units"
  fi
  echo "PROGRESS $n/5 $TOOL original $sp"
done
echo "$(date -u +%FT%TZ) queue done"
