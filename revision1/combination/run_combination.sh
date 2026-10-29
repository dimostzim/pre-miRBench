#!/bin/bash
# pre-miRBench + retrained MirMachine on the four held-out genomes (29 Oct 2026): combined scores (combine_scores.py),
# then genome_scale_metrics.py and genome_family_ap.py as for the predictors. pre-miRBench alone goes through the same
# two scripts as a check against the reported numbers. Outputs in RUN_DIR.
set -euo pipefail
W=/SCRATCH/dtzim01/premirbench/v3; REPO=/SCRATCH/dtzim01/premirbench/v2/repo; HERE=$W/combination_mirmachine_premirbench
O=${RUN_DIR:?}; D=$W/datasets/mirgenedb_71/dataset.csv; GS=$W/genome_scale; B=$W/results/benchmark/predictions
P=$B/premirbench_model_v3_original/predictions.csv; M=$B/mirmachine_retrained_genome_cutoffs/predictions.csv
mkdir -p "$O/predictions" "$O/metrics" "$O/family_ap"

python "$HERE/combine_scores.py" --premirbench "$P" --mirmachine "$M" --split valid --output "$O/predictions/valid.csv"
n=0
for sp in dme dan dya gga; do
  g=$W/genome_eval/premirbench_model_v3_original_$sp/predictions.csv
  python "$HERE/combine_scores.py" --premirbench "$g" --mirmachine "$W/comparators/mirmachine_retrained/comparison/$sp.genome_predictions.csv" \
    --split "genome_$sp" --output "$O/predictions/genome_$sp.csv"
  python "$REPO/pipeline/genome_scale/genome_scale_metrics.py" --candidates "$GS/$sp/candidates.csv" --loci "$GS/$sp/loci.csv" \
    --genome-predictions "$O/predictions/genome_$sp.csv" "$g" --validation-predictions "$O/predictions/valid.csv" "$P" \
    --training-families "$D" --genome-bases "$(grep folded_bases "$GS/candidates/$sp.stats.csv" | cut -d, -f2)" \
    --output "$O/metrics/$sp.csv"
  for method in premirbench_mirmachine_combined:"$O/predictions/genome_$sp.csv" premirbench_model_v3:"$g"; do
    python "$W/results/scripts/genome_family_ap.py" --candidates "$GS/$sp/candidates.csv" --predictions "${method#*:}" \
      --training-families "$D" --method "${method%%:*}" --genome "$sp" --output "$O/family_ap/${method%%:*}_$sp.csv"
  done
  n=$((n + 1)); echo "PROGRESS $n/4 $sp"
done
