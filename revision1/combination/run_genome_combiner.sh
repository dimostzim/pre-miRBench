#!/bin/bash
# Combiner fitted on the genome-scale validation set (29 Oct 2026; STATUS.md 6a): tables (genome_validation_set.py),
# fit and weighted maximum-F1 thresholds (combiner.py), then the four held-out genomes with genome_scale_metrics.py
# --thresholds (the combiner, and pre-miRBench alone at its genome-fitted threshold) and genome_family_ap.py.
# Needs the finished run_genome_validation.sh. Outputs in RUN_DIR.
set -euo pipefail
W=/SCRATCH/dtzim01/premirbench/v3; REPO=/SCRATCH/dtzim01/premirbench/v2/repo; HERE=$W/combination_mirmachine_premirbench
GV=$HERE/genome_validation; O=${RUN_DIR:?}; D=$W/datasets/mirgenedb_71/dataset.csv; GS=$W/genome_scale
P=$W/results/benchmark/predictions/premirbench_model_v3_original/predictions.csv
SPECIES=${SPECIES:-"cbr cel ava dgr egr bko mal gmo bpl bla loc llo aga cin cmi"}
TOOL=premirbench_mirmachine_combiner_genome
mkdir -p "$O/tables" "$O/genome/predictions" "$O/genome/metrics" "$O/genome/family_ap"

for sp in $SPECIES; do
  python "$HERE/genome_validation_set.py" --species "$sp" --candidates "$GV/$sp/candidates.csv" --loci "$GV/$sp/loci.csv" --dataset "$D" \
    --premirbench "$GV/premirbench/$sp/predictions.csv" --mirmachine "$GV/mirmachine/$sp.genome_predictions.csv" --output "$O/tables/$sp.csv"
done
tables=(); for sp in $SPECIES; do tables+=("$O/tables/$sp.csv"); done
python "$HERE/combiner.py" fit --tables "${tables[@]}" --output "$O/genome_model.json"
python "$HERE/combiner.py" threshold --model "$O/genome_model.json" --tables "${tables[@]}" --combiner-tool "$TOOL" \
  --premirbench-tool premirbench_model_v3 --output "$O/thresholds.csv"
echo "PROGRESS 1/5 fit"

n=1
for sp in dme dan dya gga; do
  g=$W/genome_eval/premirbench_model_v3_original_$sp/predictions.csv; c=$O/genome/predictions/$sp.csv
  python "$HERE/combiner.py" apply --model "$O/genome_model.json" --premirbench "$g" \
    --mirmachine "$W/comparators/mirmachine_retrained/comparison/$sp.genome_predictions.csv" --split "genome_$sp" --tool "$TOOL" --output "$c"
  python "$REPO/pipeline/genome_scale/genome_scale_metrics.py" --candidates "$GS/$sp/candidates.csv" --loci "$GS/$sp/loci.csv" \
    --genome-predictions "$c" "$g" --validation-predictions "$P" --thresholds "$O/thresholds.csv" --training-families "$D" \
    --genome-bases "$(grep folded_bases "$GS/candidates/$sp.stats.csv" | cut -d, -f2)" --output "$O/genome/metrics/$sp.csv"
  python "$W/results/scripts/genome_family_ap.py" --candidates "$GS/$sp/candidates.csv" --predictions "$c" --training-families "$D" \
    --method "$TOOL" --genome "$sp" --output "$O/genome/family_ap/${TOOL}_$sp.csv"
  n=$((n + 1)); echo "PROGRESS $n/5 $sp"
done
