#!/bin/bash
# v3 genome curves (24 Oct 2026), as v2's genome_pr_node4.sh, genome_family_ap_node4.sh and genome_discovery_node4.sh:
# precision-recall points, AP by family group and discovery-curve points of each method on each genome, written to
# <workspace>/results/genome/{pr_curves,family_ap,discovery_curves}/<method>_<sp>.csv. An existing output is kept.
# usage: RUN_DIR=<dir> genome_curves.sh <workspace> <python> <parallel jobs> <jobs file: method species predictions>
set -euo pipefail
W=$1; PY=$2; PARALLEL=$3; JOBS=$4
S=$W/results_scripts; O=$W/results/genome; D=$W/datasets/mirgenedb_71/dataset.csv
units=${RUN_DIR:?}/units.tsv
: > "$units"
mkdir -p "$O/pr_curves" "$O/family_ap" "$O/discovery_curves" "$RUN_DIR/logs"
export W PY S O D RUN_DIR

job() { # <kind> <method> <species> <predictions>
  local kind=$1 method=$2 sp=$3 pred=$4 out extra=()
  case $kind in
    pr_curves) script=genome_pr_points.py ;;
    family_ap) script=genome_family_ap.py; extra=(--training-families "$D") ;;
    discovery_curves) script=genome_discovery_points.py; extra=(--training-families "$D") ;;
  esac
  out=$O/$kind/${method}_$sp.csv
  if [ -s "$out" ]; then printf "%s/%s_%s\tskipped\texists\n" "$kind" "$method" "$sp" >> "$RUN_DIR/units.tsv"; return 0; fi
  if "$PY" "$S/$script" --candidates "$W/genome_scale/$sp/candidates.csv" --predictions "$pred" --method "$method" \
       --genome "$sp" "${extra[@]}" --output "$out" > "$RUN_DIR/logs/${kind}_${method}_$sp.log" 2>&1 && [ -s "$out" ]; then
    printf "%s/%s_%s\tok\t%s\n" "$kind" "$method" "$sp" "$out" >> "$RUN_DIR/units.tsv"
  else
    printf "%s/%s_%s\tfailed\tsee %s\n" "$kind" "$method" "$sp" "$RUN_DIR/logs/${kind}_${method}_$sp.log" >> "$RUN_DIR/units.tsv"
  fi
  echo "PROGRESS $(wc -l < "$RUN_DIR/units.tsv")/$TOTAL $kind $method $sp"
}
export -f job

TOTAL=$(( $(grep -c . "$JOBS") * 3 )); export TOTAL
while read -r method sp pred; do
  [ -s "$pred" ] || { echo "missing $pred"; exit 1; }
  for kind in pr_curves family_ap discovery_curves; do echo "$kind $method $sp $pred"; done
done < "$JOBS" | xargs -P "$PARALLEL" -L 1 bash -c 'job "$@"' _
failed=$(grep -c $'\tfailed\t' "$units" || true)
echo "$(date -u +%FT%TZ) done: $(grep -c . "$units") units, $failed failed"
