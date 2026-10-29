#!/bin/bash
# Run on the Mac (24 Oct 2026; user: "at the end bring the data to node 4", "yes do all"): copy the v3 outputs of nodes 5
# and 6 into node 4 /SCRATCH/dtzim01/premirbench/v3, as v2's consolidate_node5.sh and consolidate_node6.sh. Copy only.
# Node 5 items go node 5 -> Mac staging -> node 4 (node 4 cannot SSH to node 5); node 6 items are pulled by node 4 directly.
# An existing destination is never overwritten: if its md5 list equals the source's the item is skipped, else the script
# stops. After each copy the md5 lists of source and destination must be identical; they are appended to
# v3/archive_manifest.tsv (destination, source, md5). A missing source is listed and skipped. A copy interrupted by a
# network drop is listed in consolidate_inprogress.txt next to this script; a rerun resumes it instead of stopping.
# usage: consolidate_v3.sh A|B|C   (A: node 5 and the finished node 6 items; B: node 6 genome outputs, results and runs;
# C: items found only on nodes 5 and 6 in the 26 Oct survey, copied before their workspaces are deleted)
set -euo pipefail
PART=$1
S=/SCRATCH/dtzim01/premirbench-v3; V=/SCRATCH/dtzim01/premirbench/v3
KEY="ssh -i \$HOME/.ssh/id_ed25519_biogemt_internal -o BatchMode=yes"
opts=(-o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30)
stage=$(mktemp -d /tmp/pmb_v3_consolidate.XXXX)
inprogress=$(dirname "$0")/consolidate_inprogress.txt
touch "$inprogress"
missing=()

md5list() { # <host> <path> [rsync-style excludes as find -path patterns]: "md5  relative-path" lines; a file is "."
  local host=$1 path=$2; shift 2
  local prune=""
  for p in "$@"; do prune="$prune -path './$p' -prune -o"; done
  ssh "${opts[@]}" "$host" "if [ -d '$path' ]; then cd '$path' && find . $prune -type f -print0 | sort -z | xargs -0 -r md5sum; else echo \"\$(md5sum < '$path' | cut -d' ' -f1)  .\"; fi"
}

item() { # <host 5|6> <source relative to S> <destination relative to V> [excluded subpaths relative to the source]
  local host=$1 src=$S/$2 dst=$V/$3 slash="" a b; shift 3
  local excl=("$@") rex=()
  for e in "${excl[@]+"${excl[@]}"}"; do rex+=(--exclude="/$e"); done
  if ! ssh "${opts[@]}" "biogemt-$host" "test -e '$src'"; then missing+=("biogemt-$host:$src"); return 0; fi
  ssh "${opts[@]}" "biogemt-$host" "test -d '$src'" && slash=/
  a=$(md5list "biogemt-$host" "$src" "${excl[@]+"${excl[@]}"}")
  if ssh "${opts[@]}" biogemt-4 "test -e '$dst'"; then
    b=$(md5list biogemt-4 "$dst")
    [ "$a" = "$b" ] && { echo "same $dst"; return 0; }
    grep -qxF "$dst" "$inprogress" || { echo "ERROR: $dst exists and differs from biogemt-$host:$src"; exit 1; }
    echo "resuming interrupted copy $dst"
  fi
  echo "$dst" >> "$inprogress"
  ssh "${opts[@]}" biogemt-4 "mkdir -p '$(dirname "$dst")'"
  if [ "$host" = 6 ]; then
    ssh "${opts[@]}" biogemt-4 "rsync -a ${rex[*]+${rex[*]}} -e \"$KEY\" 'dtzim01@10.200.29.6:$src$slash' '$dst$slash'"
  else
    rm -rf "$stage/x"
    rsync -a "${rex[@]+"${rex[@]}"}" "biogemt-5:$src$slash" "$stage/x$slash"
    rsync -a "$stage/x$slash" "biogemt-4:$dst$slash"
    rm -rf "$stage/x"
  fi
  b=$(md5list biogemt-4 "$dst")
  [ "$a" = "$b" ] || { echo "ERROR: biogemt-$host:$src and $dst differ after copying"; diff <(echo "$a") <(echo "$b") | head; exit 1; }
  echo "$b" | while read -r sum path; do
    if [ "$path" = "." ]; then printf "%s\tbiogemt-%s:%s\t%s\n" "${dst#$V/}" "$host" "$src" "$sum"
    else printf "%s\tbiogemt-%s:%s\t%s\n" "${dst#$V/}/${path#./}" "$host" "$src/${path#./}" "$sum"; fi
  done | ssh "${opts[@]}" biogemt-4 "cat >> $V/archive_manifest.tsv"
  grep -vxF "$dst" "$inprogress" > "$inprogress.tmp" || true; mv "$inprogress.tmp" "$inprogress"
  echo "ok $host:$src -> $dst"
}

if [ "$PART" = A ]; then
  # Node 5: models, training logs and configs, evaluations, genome predictions and metrics, scripts.
  for tool in deepmir dnnpremir mirdnn mire2e; do
    item 5 models/$tool/v3_rep1 models/$tool/v3_rep1
    item 5 models/logs/train_${tool}_v3_rep1.log models/logs/train_${tool}_v3_rep1.log
    item 5 models/configs/$tool.v3_rep1.yaml models/configs/$tool.v3_rep1.yaml
  done
  item 5 models/premirbench_model_v2/v3_seed42 models/premirbench_model_v2/v3_seed42
  item 5 models/premirbench_model_v3/original models/premirbench_model_v3/original
  item 5 models/premirbench_model_v3/v3_seed42 models/premirbench_model_v3/v3_seed42
  for run in premirbench_model_v2_v3_seed42 premirbench_model_v3_v3_seed42; do
    item 5 models/logs/train_$run.log models/logs/train_$run.log
  done
  for run in deepmir_v3_rep1 dnnpremir_v3_rep1 mirdnn_v3_rep1 mire2e_v3_rep1 premirbench_model_v2_v3_seed42 \
             premirbench_model_v3_original premirbench_model_v3_v3_seed42; do
    item 5 evaluation/$run evaluation/$run
  done
  for f in $(ssh "${opts[@]}" biogemt-5 "cd $S/evaluation && ls *.log"); do item 5 evaluation/$f evaluation/$f; done
  for d in $(ssh "${opts[@]}" biogemt-5 "cd $S/genome_eval && ls -d *_dme *_dan *_dya *_gga"); do
    item 5 genome_eval/$d/predictions.csv genome_eval/$d/predictions.csv
    item 5 genome_eval/$d.log genome_eval/$d.log
  done
  for f in $(ssh "${opts[@]}" biogemt-5 "cd $S/genome_metrics && ls"); do item 5 genome_metrics/$f genome_metrics/$f; done
  item 5 runs node_runs/node5/runs
  for d in train genome results_scripts premirbench_model_v3_dev; do item 5 $d node_scripts/node5/$d; done
  # Node 6: genome-scale candidates, retrained MirMachine (without the training-species genomes, 90 GB), ncOrtho,
  # MuStARD and deepMiRGene models and evaluations.
  item 6 genome_scale genome_scale
  item 6 mirmachine comparators/mirmachine_retrained species_inputs
  item 6 comparators/ncortho comparators/ncortho
  item 6 comparators/run_ncortho_v3.sh node_scripts/node6/comparators/run_ncortho_v3.sh
  for tool in mustard deepmirgene; do
    item 6 models/$tool/v3_rep1 models/$tool/v3_rep1
    item 6 evaluation/${tool}_v3_rep1 evaluation/${tool}_v3_rep1
    item 6 models/configs/$tool.v3_rep1.yaml models/configs/$tool.v3_rep1.yaml
  done
  for f in $(ssh "${opts[@]}" biogemt-6 "cd $S/models/logs && ls"); do item 6 models/logs/$f models/logs/$f; done
  for f in $(ssh "${opts[@]}" biogemt-6 "cd $S/evaluation && ls *.log"); do item 6 evaluation/$f evaluation/$f; done
fi

if [ "$PART" = B ]; then
  for d in $(ssh "${opts[@]}" biogemt-6 "cd $S/genome_eval && ls -d mustard_??? deepmirgene_??? deepmir_gga"); do
    item 6 genome_eval/$d/predictions.csv genome_eval/$d/predictions.csv
    item 6 genome_eval/$d.log genome_eval/$d.log
  done
  for f in $(ssh "${opts[@]}" biogemt-6 "cd $S/genome_metrics && ls"); do item 6 genome_metrics/$f genome_metrics/$f; done
  item 6 results results
  item 6 runs node_runs/node6/runs
  for d in train genome results_scripts; do item 6 $d node_scripts/node6/$d; done
fi

if [ "$PART" = C ]; then
  # 26 Oct (user: "yes lets do all"): the two failed MuStARD attempts (GPU out of memory) and the run launchers.
  for run in v3_rep1_INCOMPLETE_20261023T061535Z v3_rep1_INCOMPLETE_20261023T120505Z; do
    item 6 models/mustard/$run models/mustard/$run
  done
  item 5 bin node_scripts/node5/bin
  item 6 bin node_scripts/node6/bin
  # The final v2 run copy of the code (identical on nodes 5 and 6) goes into v2, listed in v2/archive_manifest.tsv.
  # Left out: Python bytecode and results/, which holds only the tool containers' home caches (keras, CUDA, torch hub).
  S=/SCRATCH/dtzim01/premirbench-v2-training; V=/SCRATCH/dtzim01/premirbench/v2
  caches=(pipeline/__pycache__ pipeline/comparators/__pycache__ pipeline/dataset/__pycache__
          pipeline/genome_scale/__pycache__ results)
  [ "$(md5list biogemt-5 $S/repo "${caches[@]}")" = "$(md5list biogemt-6 $S/repo "${caches[@]}")" ] ||
    { echo "ERROR: the node 5 and node 6 run copies of the v2 code differ"; exit 1; }
  item 5 repo code_snapshots/v2_final_run_copy "${caches[@]}"
fi

rm -rf "$stage"
if [ ${#missing[@]} -gt 0 ]; then printf "missing: %s\n" "${missing[@]}"; fi
echo "$(date -u +%FT%TZ) part $PART done"
