#!/bin/bash
# Combiner fitted on the validation windows (29 Oct 2026; STATUS.md 6a): fit (combiner.py), scores on the validation
# partition and Tests 1-4, summarize_benchmark.py for pre-miRBench, MirMachine and the combiner (AP, 95% bootstrap CI,
# paired differences), then the same combiner on the four held-out genomes with genome_scale_metrics.py (thresholds
# from the validation windows) and genome_family_ap.py. Outputs in RUN_DIR.
set -euo pipefail
W=/SCRATCH/dtzim01/premirbench/v3; REPO=/SCRATCH/dtzim01/premirbench/v2/repo; HERE=$W/combination_mirmachine_premirbench
O=${RUN_DIR:?}; D=$W/datasets/mirgenedb_71/dataset.csv; GS=$W/genome_scale; B=$W/results/benchmark/predictions
P=$B/premirbench_model_v3_original/predictions.csv; M=$B/mirmachine_retrained_genome_cutoffs/predictions.csv
TOOL=premirbench_mirmachine_combiner_window
mkdir -p "$O/tables" "$O/benchmark/$TOOL" "$O/genome/predictions" "$O/genome/metrics" "$O/genome/family_ap"

python "$HERE/combiner.py" table --premirbench "$P" --mirmachine "$M" --split valid --output "$O/tables/valid_windows.csv"
python "$HERE/combiner.py" fit --tables "$O/tables/valid_windows.csv" --output "$O/window_model.json"
out=$O/benchmark/$TOOL/predictions.csv
for split in valid test_known_species_known_family test_known_species_heldout_family test_heldout_species_known_family \
             test_heldout_species_heldout_family; do
  python "$HERE/combiner.py" apply --model "$O/window_model.json" --premirbench "$P" --mirmachine "$M" --split "$split" \
    --tool "$TOOL" --output "$O/benchmark/$split.csv"
  if [ "$split" = valid ]; then cp "$O/benchmark/$split.csv" "$out"; else tail -n +2 "$O/benchmark/$split.csv" >> "$out"; fi
done
python "$REPO/pipeline/summarize_benchmark.py" --predictions "$P" "$M" "$out" --output-dir "$O/benchmark/summary"
echo "PROGRESS 1/5 benchmark"

n=1
for sp in dme dan dya gga; do
  g=$W/genome_eval/premirbench_model_v3_original_$sp/predictions.csv; c=$O/genome/predictions/$sp.csv
  python "$HERE/combiner.py" apply --model "$O/window_model.json" --premirbench "$g" \
    --mirmachine "$W/comparators/mirmachine_retrained/comparison/$sp.genome_predictions.csv" --split "genome_$sp" --tool "$TOOL" --output "$c"
  python "$REPO/pipeline/genome_scale/genome_scale_metrics.py" --candidates "$GS/$sp/candidates.csv" --loci "$GS/$sp/loci.csv" \
    --genome-predictions "$c" --validation-predictions "$out" --training-families "$D" \
    --genome-bases "$(grep folded_bases "$GS/candidates/$sp.stats.csv" | cut -d, -f2)" --output "$O/genome/metrics/$sp.csv"
  python "$W/results/scripts/genome_family_ap.py" --candidates "$GS/$sp/candidates.csv" --predictions "$c" --training-families "$D" \
    --method "$TOOL" --genome "$sp" --output "$O/genome/family_ap/${TOOL}_$sp.csv"
  n=$((n + 1)); echo "PROGRESS $n/5 $sp"
done
