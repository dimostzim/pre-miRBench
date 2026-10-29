#!/bin/bash
# v3 genome-scale candidates for the held-out genomes: extract_hairpin_candidates.py --mode full (repeat rule on the
# hairpin, no window shift, no exclusions), then labels, loci and predictor inputs (prepare_genome_candidates.py).
# Conda env premirbench-cm (ViennaRNA 2.7.2, as the node 4 build env). Writes units.tsv in the run directory ($RUN_DIR).
set -euo pipefail
export PATH=/SCRATCH/dtzim01/conda-envs/premirbench-cm/bin:$PATH  # env premirbench-cm without activation
W=/SCRATCH/dtzim01/premirbench-v3
GS=$W/genome_scale
CPUS=${CPUS:-88}
mkdir -p "$GS/candidates"
units=${RUN_DIR:?}/units.tsv
: > "$units"
n=0
for sp in dme dan dya gga; do
  echo "$(date -u +%FT%TZ) candidates $sp"
  python -I "$W/repo/pipeline/dataset/extract_hairpin_candidates.py" --genome "$GS/inputs/$sp.genome.prefixed.fa" --mode full \
    --cpus "$CPUS" --output "$GS/candidates/$sp.candidates.csv" --stats-output "$GS/candidates/$sp.stats.csv"
  mkdir -p "$GS/$sp"
  # MuStARD reads sequences from <dataset-dir>/genome.fa; a hard link avoids a second copy.
  ln -f "$GS/inputs/$sp.genome.prefixed.fa" "$GS/$sp/genome.fa"
  echo "$(date -u +%FT%TZ) labels and inputs $sp"
  python -I "$W/repo/pipeline/genome_scale/prepare_genome_candidates.py" --candidates "$GS/candidates/$sp.candidates.csv" \
    --bed "$GS/inputs/$sp.precursors.prefixed.bed" --genome "$GS/inputs/$sp.genome.prefixed.fa" --species "$sp" --output-dir "$GS/$sp"
  [ -s "$GS/$sp/candidates.csv" ] && [ -s "$GS/$sp/loci.csv" ]
  printf "%s\tok\t\n" "$sp" >> "$units"
  n=$((n + 1))
  echo "PROGRESS $n/4 $sp done"
done
