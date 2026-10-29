#!/bin/bash
# v3 Agentomics run: the v2 run 2 command (code_changes.md, Agentomics) with the v3 export and a new workspace.
set -o pipefail
cd ~/agentomics-ml
~/agentomics-ml/.venv/bin/agentomics-run --image agentomics:501a27d-v4 --model gpt-6.1-sol --provider codex --dataset premirbench_v3 --datasets-dir /SCRATCH/dtzim01/premirbench/v3/agentomics/datasets --workspace-dir /SCRATCH/dtzim01/premirbench/v3/agentomics/workspace/premirbench_v3_run1 --iterations 20 --val-metric AUPRC --split-allowed-iterations 0 --task-type classification --conda-export-mode full --verbosity full --user-prompt "Develop a machine learning model that generalizes well to new unseen data. Do not stack or ensemble more than three models." 2>&1 | tee /SCRATCH/dtzim01/premirbench/v3/agentomics/logs/premirbench_v3_run1.log
