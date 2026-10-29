#!/bin/bash
# CPU checks of pre-miRBench_model_v3 on node 5 (23-24 Oct 2026):
# 1. 2-epoch smoke runs on a 600-row training subset: our train.py against the agent's iteration-19, -18
#    and -11 scripts (seeds 42, 73, 17), compared tensor by tensor;
# 2. the agent's original model packaged with package_original.py: our inference.py against the agent's
#    inference.py on the full validation split, on CPU.
set -euo pipefail
W=/SCRATCH/dtzim01/premirbench-v3
PKG=$W/repo/pre-miRBench_model_v3
SRC=$W/agentomics/original_v3
DATA=$W/agentomics/datasets/premirbench_v3
DEV=$W/premirbench_model_v3_dev
PY=/SCRATCH/dtzim01/conda-envs/premirbench_model_v2/bin/python
export TMPDIR=$DEV/tmp PYTHONDONTWRITEBYTECODE=1
OUT=$DEV/check_cpu
[ -e "$OUT" ] && { echo "$OUT exists"; exit 1; }
mkdir -p "$TMPDIR" "$OUT"

echo "$(date -u +%FT%TZ) mini splits"
"$PY" - "$DATA" "$OUT/mini" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
data, out = Path(sys.argv[1]), Path(sys.argv[2])
for split, rows in (("train", 600), ("validation", 300)):
    samples = pd.read_csv(data / split / "input" / "samples.tsv", sep="\t", dtype={"id": str}).head(rows)
    labels = pd.read_csv(data / split / "labels.csv", dtype={"id": str}).set_index("id").loc[samples.id].reset_index()
    (out / split / "input").mkdir(parents=True)
    samples.to_csv(out / split / "input" / "samples.tsv", sep="\t", index=False)
    labels.rename(columns={"label": "numeric_label"}).to_csv(out / split / "labels.csv", index=False)
    print(split, len(samples), int(labels.label.sum()), "positives")
EOF

for i in 19 18 11; do
  echo "$(date -u +%FT%TZ) agent iteration $i smoke run (CPU)"
  cp -r "$SRC/run/iteration_$i/model_training" "$OUT/agent_i$i"
  (cd "$OUT/agent_i$i" && CUDA_VISIBLE_DEVICES= "$PY" -u train.py --train-data "$OUT/mini/train" --validation-data "$OUT/mini/validation" \
     --artifacts-dir "$OUT/agent_i$i/artifacts" > "$OUT/agent_i$i.log" 2>&1) || echo "agent iteration $i exited non-zero (see log; checkpoints are written during training)"
done

echo "$(date -u +%FT%TZ) our train.py smoke run (CPU)"
CUDA_VISIBLE_DEVICES= "$PY" -u "$PKG/train.py" --train-data "$OUT/mini/train" --validation-data "$OUT/mini/validation" \
  --artifacts-dir "$OUT/ours" --device cpu > "$OUT/ours.log" 2>&1
tail -1 "$OUT/ours.log"
status=0
"$PY" "$DEV/compare_states.py" "$OUT/ours/aligned_network_1.pt" "$OUT/agent_i19/artifacts/component_seed_42.pt" "aligned network 1 vs iteration 19 (seed 42)" || status=1
"$PY" "$DEV/compare_states.py" "$OUT/ours/aligned_network_2.pt" "$OUT/agent_i18/artifacts/component_seed_73.pt" "aligned network 2 vs iteration 18 (seed 73)" || status=1
"$PY" "$DEV/compare_states.py" "$OUT/ours/folded_grid_network.pt" "$OUT/agent_i11/artifacts/component_seed_17.pt" "folded-grid network vs iteration 11 (seed 17)" || status=1

echo "$(date -u +%FT%TZ) package the original model; validation inference on CPU, ours and the agent's"
(cd "$PKG" && "$PY" package_original.py --source "$SRC/best_iteration_snapshot/model_training/training_artifacts" --artifacts-dir "$OUT/original_bundle")
CUDA_VISIBLE_DEVICES= "$PY" "$PKG/inference.py" --input "$DATA/validation/input" --output "$OUT/validation_ours_cpu.csv" \
  --artifacts-dir "$OUT/original_bundle" --device cpu
(cd "$SRC/best_iteration_snapshot/model_training/training_artifacts" && CUDA_VISIBLE_DEVICES= "$PY" inference.py \
  --input "$DATA/validation/input" --output "$OUT/validation_agent_cpu.csv" --artifacts-dir .)
if cmp -s "$OUT/validation_ours_cpu.csv" "$OUT/validation_agent_cpu.csv"; then echo "validation CPU predictions: byte-identical"; else echo "validation CPU predictions: DIFFER"; status=1; fi
"$PY" - "$DATA/validation/labels.csv" "$OUT/validation_ours_cpu.csv" "$SRC/best_iteration_snapshot/validation_evaluation/eval_predictions_validation.csv" <<'EOF'
import sys
import pandas as pd
from sklearn.metrics import average_precision_score
labels = pd.read_csv(sys.argv[1], dtype={"id": str})
for name, path in (("ours, CPU", sys.argv[2]), ("Agentomics evaluation", sys.argv[3])):
    merged = labels.merge(pd.read_csv(path, dtype={"id": str}), on="id", validate="one_to_one")
    print(f"validation AP {name}: {average_precision_score(merged.label, merged.probability_1):.16f} ({len(merged)} rows)")
EOF
rm -rf "$TMPDIR"
echo "$(date -u +%FT%TZ) done, status $status"
exit $status
