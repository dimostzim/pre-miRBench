#!/usr/bin/env bash

set -euo pipefail

prefix="balanced_benchmark"
prefix_set=0
norm_output="${NORM_OUTPUT:-n}"
tools=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --norm-output|--norm_output)
      norm_output="${2:?missing value for $1}"
      shift 2
      ;;
    *)
      if [[ "$prefix_set" -eq 0 ]]; then
        prefix="$1"
        prefix_set=1
      else
        tools+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ "${#tools[@]}" -eq 0 ]]; then
  tools=(deepmir deepmirgene dnnpremir mirdnn mire2e mustard)
fi

for tool in "${tools[@]}"; do
  cmd=(python tools/inference.py \
    --tool "$tool" \
    --output-name "$prefix" \
    --config "benchmark/balanced_benchmark/configs/${tool}.yaml")
  if [[ "$norm_output" == "y" ]]; then
    cmd+=(--norm-output y)
  fi
  "${cmd[@]}"
done
