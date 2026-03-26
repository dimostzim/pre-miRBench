#!/usr/bin/env bash

set -euo pipefail

prefix="${1:-1_1_collapsed}"
shift || true

if [[ $# -gt 0 ]]; then
  tools=("$@")
else
  tools=(deepmir deepmirgene dnnpremir mirdnn mire2e mustard)
fi

for tool in "${tools[@]}"; do
  python tools/inference.py \
    --tool "$tool" \
    --output-name "$prefix" \
    --config "benchmark/1_1/configs/${tool}.yaml"
done
