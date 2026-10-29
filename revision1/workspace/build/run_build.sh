#!/bin/bash
# pre-miRBench v3 build: v2 settings plus loop-centred positives, the repeat rule on the hairpin and a seeded
# window shift of at most 10 nt for every window. Usage: run_build.sh OUT WORK [extra build_dataset.py args]
# The revision1 code is not committed yet, so the exact source state is recorded as HEAD + diff + untracked files.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh

REPO=/SCRATCH/dtzim01/premirbench/v2/repo
V2=/SCRATCH/dtzim01/premirbench/v2
OUT=$1
WORK=$2
shift 2
PROV=${OUT}_provenance

mkdir -p "$PROV" "$OUT" "$WORK"
CMD=(python -u "$REPO/pipeline/build_dataset.py"
     --panel "$V2/raw/mirgenedb_71/panel.tsv"
     --output-dir "$OUT"
     --work-dir "$WORK"
     --ratio 10
     --negative-sample-bases 20000000
     --cpus 8
     --species-jobs 11
     --heldout-species gga,dme,dan,dya
     --max-shift 10
     "$@")

{
  echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo "git_head: $(git -C "$REPO" rev-parse HEAD)"
  echo "git_branch: $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  printf "command:"; printf " %q" "${CMD[@]}"; echo
} > "$PROV/build_metadata.txt"
git -C "$REPO" status --porcelain > "$PROV/git_status.txt"
git -C "$REPO" diff HEAD > "$PROV/git_diff_head.patch"
git -C "$REPO" ls-files --others --exclude-standard -z | grep -zv __pycache__ | tar -C "$REPO" --null -T - -czf "$PROV/untracked_files.tar.gz"
conda env export -n premirbench > "$PROV/conda_premirbench.yml"
cp "$V2/raw/mirgenedb_71/panel.tsv" "$PROV/panel.tsv"

cd "$REPO"
"${CMD[@]}"
echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$PROV/build_metadata.txt"
