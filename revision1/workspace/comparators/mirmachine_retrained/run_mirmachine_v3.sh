#!/bin/bash
# Retrained MirMachine on the v3 training partition (node 6), as for v2 (methods_revision1.md 6.1): family models and
# window-fitted cutoffs, species inputs and genome-fitted cutoffs on the training genomes, benchmark scores with the
# genome-fitted cutoffs (families without a fitting hit keep the window-fitted cutoff; user decision 22 Oct, option a),
# then the four held-out genomes, compared with the v3 loci and candidates. Writes units.tsv in $RUN_DIR.
# At most about 48 threads, so training and evaluation on node 6 keep the other half of the CPUs.
set -euo pipefail
W=/SCRATCH/dtzim01/premirbench-v3
M=$W/mirmachine
CODE=$W/repo/pipeline/comparators
D=$W/datasets/mirgenedb_71/dataset.csv
E=/SCRATCH/dtzim01/conda-envs/mirmachine
CMBIN=/SCRATCH/dtzim01/conda-envs/premirbench-cm/bin
REVIEW=/SCRATCH/dtzim01/conda-envs/premirbench-review/bin/python
META=$E/lib/python3.13/site-packages/mirmachine/meta
TOOL=mirmachine_retrained_genome_cutoffs
SPLITS=valid,test_known_species_known_family,test_known_species_heldout_family,test_heldout_species_known_family,test_heldout_species_heldout_family
units=${RUN_DIR:?}/units.tsv
: > "$units"
n=0
done_unit() { n=$((n + 1)); printf "%s\tok\t\n" "$1" >> "$units"; echo "PROGRESS $n/9 $1"; }
mkdir -p "$M/window" "$M/genome" "$M/comparison"

echo "$(date -u +%FT%TZ) build"
PATH=$CMBIN:$PATH python -u "$CODE/mirmachine_retrained.py" build --dataset "$D" --work-dir "$M/window" --cpus 48
done_unit build
echo "$(date -u +%FT%TZ) window-fitted cutoffs"
PATH=$CMBIN:$PATH python -u "$CODE/mirmachine_retrained.py" cutoffs --dataset "$D" --work-dir "$M/window" --cpus 48
done_unit window-cutoffs
echo "$(date -u +%FT%TZ) species inputs"
PATH=$E/bin:$PATH "$E/bin/python" -u "$CODE/mirmachine_species_inputs.py" --dataset "$D" --panel "$M/inputs/panel.tsv" \
  --mirmachine "$E/bin/MirMachine.py" --tree "$META/tree.newick" --genome "$W/datasets/mirgenedb_71/genome.fa" \
  --bed-dir "$M/inputs/bed" --output-dir "$M/species_inputs"
done_unit species-inputs
ln -f "$M/window/library.cm" "$M/genome/library.cm"
cp "$M/window/families.tsv" "$M/genome/families.tsv"
echo "$(date -u +%FT%TZ) genome-fitted cutoffs"
PATH=$CMBIN:$PATH python -u "$CODE/mirmachine_retrained.py" genome-cutoffs --work-dir "$M/genome" --dataset "$D" \
  --species-table "$M/species_inputs/species_table.tsv" --cpus 16 --parallel 3
done_unit genome-cutoffs
echo "$(date -u +%FT%TZ) score benchmark"
PATH=$CMBIN:$PATH python -u "$CODE/mirmachine_retrained.py" score --work-dir "$M/genome" --dataset "$D" --cpus 24 --name benchmark \
  --tool-name "$TOOL" --splits "$SPLITS" --fallback-work-dir "$M/window" --output "$M/benchmark_predictions.csv"
[ -s "$M/benchmark_predictions.csv" ]
done_unit benchmark
declare -A NAME=([dme]=Drosophila_melanogaster [dan]=Drosophila_ananassae [dya]=Drosophila_yakuba [gga]=Gallus_gallus)
for sp in dme dan dya gga; do
  echo "$(date -u +%FT%TZ) genome $sp"
  PATH=$CMBIN:$PATH python -u "$CODE/mirmachine_retrained.py" genome --work-dir "$M/genome" --cpus 24 --name "$sp" \
    --genome "$W/genome_scale/inputs/$sp.genome.prefixed.fa" --mirmachine-gff "$M/inputs/mirmachine_gff/${NAME[$sp]}.PRE.gff" \
    --all-gff "$M/genome_$sp.all.gff" --filtered-gff "$M/genome_$sp.filtered.gff"
  "$REVIEW" "$CODE/compare_annotation_calls.py" --tool "$TOOL" --format mirmachine_gff \
    --calls "$M/genome_$sp.all.gff" --called "$M/genome_$sp.filtered.gff" --loci "$W/genome_scale/$sp/loci.csv" \
    --training-families "$D" --genome-bases "$(grep folded_bases "$W/genome_scale/candidates/$sp.stats.csv" | cut -d, -f2)" \
    --candidates "$W/genome_scale/$sp/candidates.csv" \
    --summary "$M/comparison/$sp.summary.csv" --predictions "$M/comparison/$sp.genome_predictions.csv"
  [ -s "$M/comparison/$sp.summary.csv" ]
  done_unit "genome-$sp"
done
echo "$(date -u +%FT%TZ) done"
