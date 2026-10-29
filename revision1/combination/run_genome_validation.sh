#!/bin/bash
# Genome-scale validation set for the pre-miRBench + MirMachine combiner (29 Oct 2026). For each training species:
# candidates as for the held-out genomes (genome_scale/run_candidates.sh), pre-miRBench scores (as
# node_scripts/node5/train/premirbench_model_v3_original_queue.sh) and MirMachine's per-candidate scores from its own
# cutoff-fitting search of the genome (mirmachine_training_genome_gff.py, then compare_annotation_calls.py as for the
# held-out genomes). A finished step is kept. Writes units.tsv in RUN_DIR.
set -euo pipefail
W=/SCRATCH/dtzim01/premirbench/v3; REPO=/SCRATCH/dtzim01/premirbench/v2/repo; HERE=$W/combination_mirmachine_premirbench
GV=$HERE/genome_validation; B=$W/work/build_mirgenedb_71; D=$W/datasets/mirgenedb_71/dataset.csv
MM=$W/comparators/mirmachine_retrained/genome
VIENNA=/SCRATCH/dtzim01/conda-envs/premirbench/bin  # RNALfold 2.7.2, the version used for the held-out genomes
PY=/SCRATCH/dtzim01/conda-envs/premirbench_model_v2/bin/python
RPY=/SCRATCH/dtzim01/conda-envs/premirbench-review/bin/python
SPECIES=${SPECIES:-"cbr cel ava dgr egr bko mal gmo bpl bla loc llo aga cin cmi"}
units=${RUN_DIR:?}/units.tsv
: > "$units"
export TMPDIR=$GV/tmp XDG_CACHE_HOME=$GV/tmp/cache PYTHONDONTWRITEBYTECODE=1
mkdir -p "$GV/candidates" "$GV/premirbench" "$GV/mirmachine" "$TMPDIR" "$XDG_CACHE_HOME"
total=$(wc -w <<< "$SPECIES"); n=0
for sp in $SPECIES; do
  d=$GV/$sp; g=$GV/premirbench/$sp; m=$GV/mirmachine/$sp
  if [ ! -s "$d/candidates.csv" ] || [ ! -s "$d/loci.csv" ]; then
    echo "$(date -u +%FT%TZ) candidates $sp"
    PATH=$VIENNA:$PATH python -I "$REPO/pipeline/dataset/extract_hairpin_candidates.py" --genome "$B/$sp/genome.prefixed.fa" --mode full \
      --cpus 88 --output "$GV/candidates/$sp.candidates.csv" --stats-output "$GV/candidates/$sp.stats.csv"
    mkdir -p "$d"
    echo "$(date -u +%FT%TZ) labels and inputs $sp"
    PATH=$VIENNA:$PATH python -I "$REPO/pipeline/genome_scale/prepare_genome_candidates.py" --candidates "$GV/candidates/$sp.candidates.csv" \
      --bed "$B/$sp/precursors.prefixed.bed" --genome "$B/$sp/genome.prefixed.fa" --species "$sp" --output-dir "$d"
  fi
  if [ ! -s "$g/predictions.csv" ]; then
    echo "$(date -u +%FT%TZ) pre-miRBench $sp"
    "$PY" -u "$REPO/pipeline/genome_scale/run_premirbench_model_genome.py" --repo "$REPO" --genome-dataset-dir "$d" --species "$sp" \
      --artifacts-dir "$W/models/premirbench_model_v3/original" --python "$PY" --output-dir "$g" --parallel 22 --cores-per-chunk 4 \
      --model-dir "$REPO/pre-miRBench_model_v3" --tool premirbench_model_v3 > "$g.log" 2>&1
  fi
  if [ ! -s "$m.genome_predictions.csv" ]; then
    echo "$(date -u +%FT%TZ) MirMachine $sp"
    "$RPY" "$HERE/mirmachine_training_genome_gff.py" --table "$MM/genome_cutoffs/$sp.tbl" --work-dir "$MM" --all-gff "$m.all.gff" \
      --filtered-gff "$m.filtered.gff"
    "$RPY" "$REPO/pipeline/comparators/compare_annotation_calls.py" --tool mirmachine_retrained_genome_cutoffs --calls "$m.all.gff" \
      --called "$m.filtered.gff" --format mirmachine_gff --loci "$d/loci.csv" --training-families "$D" \
      --genome-bases "$(grep folded_bases "$GV/candidates/$sp.stats.csv" | cut -d, -f2)" --candidates "$d/candidates.csv" \
      --summary "$m.summary.csv" --predictions "$m.genome_predictions.csv" > "$m.log" 2>&1
  fi
  if [ -s "$g/predictions.csv" ] && [ -s "$m.genome_predictions.csv" ]; then
    printf "%s\tok\t%s\n" "$sp" "$d" >> "$units"
  else
    printf "%s\tfailed\tsee %s.log and %s.log\n" "$sp" "$g" "$m" >> "$units"
  fi
  n=$((n + 1)); echo "PROGRESS $n/$total $sp"
done
echo "$(date -u +%FT%TZ) done"
