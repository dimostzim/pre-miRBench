#!/usr/bin/env bash

set -euo pipefail

prefix="${1:-scan_chr14}"
shift || true

if [[ $# -gt 0 ]]; then
  tools=("$@")
else
  tools=(deepmir deepmirgene dnnpremir mirdnn mire2e mustard)
fi

for tool in "${tools[@]}"; do
  config_dir="benchmark/prepared_inputs/scan_benchmark/${tool}/configs"
  if [[ ! -d "$config_dir" ]]; then
    echo "Missing prepared scan configs for ${tool}: ${config_dir}" >&2
    exit 1
  fi

  mapfile -t configs < <(find "$config_dir" -maxdepth 1 -type f -name '*.yaml' | sort)
  if [[ ${#configs[@]} -eq 0 ]]; then
    echo "No scan configs found for ${tool} under ${config_dir}" >&2
    exit 1
  fi

  for config in "${configs[@]}"; do
    chunk_id="$(basename "${config%.yaml}")"
    python tools/inference.py \
      --tool "$tool" \
      --output-name "${prefix}/${chunk_id}" \
      --config "$config"
  done
done
