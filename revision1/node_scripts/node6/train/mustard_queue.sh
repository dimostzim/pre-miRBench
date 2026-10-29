#!/bin/bash
# v3 MuStARD rerun after the GPU out-of-memory failure of 22 Oct (epoch 279): train once more with the same settings
# (seed 1, run v3_rep1; the failed folder is renamed *_INCOMPLETE_* by train_replicates.sh), apply the checks of
# train_queue.sh, then evaluate the five splits in parallel, one evaluate.py per split with the arguments of
# evaluate_replicates.sh (as for v2 on 20 Oct), and merge them into evaluation/mustard_v3_rep1/predictions.csv.
# Genome scoring runs as a separate queue (genome_queue_species.sh with SCORE_FIRST=1) once units.tsv has "train ok".
# usage: RUN_DIR=<dir> mustard_queue.sh <workspace> <repo>
set -euo pipefail
W=$1; REPO=$2
D=$W/datasets/mirgenedb_71
export RUN_PREFIX=v3
tool=mustard; run=v3_rep1
units=${RUN_DIR:?}/units.tsv
: > "$units"
log=$W/models/logs/train_${tool}_${run}.log
echo "$(date -u +%FT%TZ) train $tool $run"
bash "$REPO/pipeline/train_replicates.sh" "$D" "$W/models" "$tool" 1 || true
if [ ! -f "$W/models/$tool/$run/inference_config.yaml" ] || grep -q "^Traceback (most recent call last)" "$log" || ! grep -aqE "val_auprc: " "$log"; then
  printf "train\tfailed\tno inference config, a traceback or no validation-AUPRC selection line in %s\n" "$log" >> "$units"
  echo "PROGRESS 1/2 train FAILED"; exit 1
fi
printf "train\tok\t%s\n" "$log" >> "$units"; echo "PROGRESS 1/2 trained $tool"
EV=$W/evaluation/${tool}_${run}
[ -e "$EV" ] && mv "$EV" "${EV}_INCOMPLETE_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$EV/splits"
pids=()
for split in valid test_known_species_known_family test_known_species_heldout_family test_heldout_species_known_family test_heldout_species_heldout_family; do
  echo "$(date -u +%FT%TZ) evaluate $tool $run $split"
  python3 -u "$REPO/pipeline/evaluate.py" --dataset-dir "$D" --training-root "$W/models" --run-name "$run" \
    --tools "$tool" --splits "$split" --output-dir "$EV/splits/$split" --cpu-only --cpu-threads 8 > "$EV/splits/$split.log" 2>&1 &
  pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
files=("$EV"/splits/*/predictions.csv)
if [ "$failed" = 1 ] || [ "${#files[@]}" -ne 5 ] || ! [ -s "${files[0]}" ]; then
  printf "evaluation\tfailed\tsee %s/splits/<split>.log\n" "$EV" >> "$units"; echo "PROGRESS 2/2 evaluation FAILED"; exit 1
fi
# Same header in all five files; written under a temporary name and moved, so a non-empty predictions.csv is complete.
{ head -n 1 "${files[0]}"; for f in "${files[@]}"; do tail -n +2 "$f"; done; } > "$EV/predictions.csv.tmp"
mv "$EV/predictions.csv.tmp" "$EV/predictions.csv"
printf "evaluation\tok\t%s\n" "$EV/predictions.csv" >> "$units"; echo "PROGRESS 2/2 evaluated $tool"
echo "$(date -u +%FT%TZ) done"
