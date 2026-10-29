#!/bin/bash
# ncOrtho (chicken) against the v3 loci and candidates, on node 6. Same ncOrtho run as v2 (19 Oct, all 573 human
# references; calls copied from node 4 v2/comparators/results/ncortho_gga.prefixed.tsv); only predictions from human
# references whose family is a v3 training family are counted (methods_revision1.md 6.2; ncOrtho searches each
# reference independently, so this equals a rerun without the other references).
set -euo pipefail
W=/SCRATCH/dtzim01/premirbench-v3
PY=/SCRATCH/dtzim01/conda-envs/premirbench-review/bin/python
D=$W/datasets/mirgenedb_71/dataset.csv
GS=$W/genome_scale
OUT=$W/comparators/ncortho
mkdir -p "$OUT"
"$PY" - "$W/mirmachine/inputs/ncortho_gga.prefixed.tsv" "$D" "$OUT/ncortho_gga_training_family_refs.prefixed.tsv" <<'PY'
import csv
import sys
sys.path.insert(0, "/SCRATCH/dtzim01/premirbench-v3/repo/pipeline")
from build_dataset import family_id_from_mirna_id
calls, dataset, output = sys.argv[1:]
training = {row["family_id"] for row in csv.DictReader(open(dataset)) if row["split"] == "train" and row["label"] == "1"}
rows = list(csv.DictReader(open(calls), delimiter="\t"))
kept = [row for row in rows if family_id_from_mirna_id(row["human_mirna_id"]) in training]
with open(output, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(kept)
print(f"{len(kept)} of {len(rows)} ncOrtho predictions come from {len({r['human_mirna_id'] for r in kept})} training-family references")
PY
"$PY" -u "$W/repo/pipeline/comparators/compare_annotation_calls.py" --tool ncortho_training_family_refs --format ncortho_tsv \
  --calls "$OUT/ncortho_gga_training_family_refs.prefixed.tsv" --loci "$GS/gga/loci.csv" --training-families "$D" \
  --genome-bases "$(grep folded_bases "$GS/candidates/gga.stats.csv" | cut -d, -f2)" --candidates "$GS/gga/candidates.csv" \
  --summary "$OUT/ncortho_training_family_refs_gga.summary.csv" --predictions "$OUT/ncortho_training_family_refs_gga.genome_predictions.csv"
