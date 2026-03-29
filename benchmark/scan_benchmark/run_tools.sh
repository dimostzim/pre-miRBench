#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash benchmark/scan_benchmark/run_tools.sh [options] [prefix] [tool ...]

Options:
  --jobs-default N      Default parallel chunk workers for tools without an override.
  --jobs SPEC           Comma-separated per-tool overrides, for example:
                        deepmir=2,deepmirgene=2,mirdnn=1
  -h, --help            Show this help.

Defaults:
  deepmir=2, deepmirgene=2, dnnpremir=1, mirdnn=1, mire2e=1, mustard=1
EOF
}

all_tools=(deepmir deepmirgene dnnpremir mirdnn mire2e mustard)
jobs_default=1
jobs_default_set=0

declare -A recommended_jobs=(
  [deepmir]=2
  [deepmirgene]=2
  [dnnpremir]=1
  [mirdnn]=1
  [mire2e]=1
  [mustard]=1
)
declare -A job_overrides=()

is_known_tool() {
  local candidate="$1"
  local tool
  for tool in "${all_tools[@]}"; do
    if [[ "$tool" == "$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

require_positive_int() {
  local value="$1"
  local label="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid ${label}: ${value}" >&2
    exit 1
  fi
}

parse_job_overrides() {
  local spec="$1"
  local entry tool jobs
  IFS=',' read -r -a entries <<< "$spec"
  for entry in "${entries[@]}"; do
    [[ -n "$entry" ]] || continue
    if [[ "$entry" != *=* ]]; then
      echo "Invalid --jobs entry: ${entry}" >&2
      exit 1
    fi
    tool="${entry%%=*}"
    jobs="${entry#*=}"
    if ! is_known_tool "$tool"; then
      echo "Unknown tool in --jobs: ${tool}" >&2
      exit 1
    fi
    require_positive_int "$jobs" "job count for ${tool}"
    job_overrides["$tool"]="$jobs"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs-default)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --jobs-default" >&2; exit 1; }
      require_positive_int "$1" "--jobs-default"
      jobs_default="$1"
      jobs_default_set=1
      shift
      ;;
    --jobs-default=*)
      jobs_default="${1#*=}"
      require_positive_int "$jobs_default" "--jobs-default"
      jobs_default_set=1
      shift
      ;;
    --jobs)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --jobs" >&2; exit 1; }
      parse_job_overrides "$1"
      shift
      ;;
    --jobs=*)
      parse_job_overrides "${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

prefix="${1:-scan_chr14}"
shift || true

if [[ $# -gt 0 ]]; then
  tools=("$@")
else
  tools=("${all_tools[@]}")
fi

run_chunk() {
  local tool="$1"
  local prefix="$2"
  local config="$3"
  local chunk_id
  chunk_id="$(basename "${config%.yaml}")"
  python tools/inference.py \
    --tool "$tool" \
    --output-name "${prefix}/${chunk_id}" \
    --config "$config"
}

for tool in "${tools[@]}"; do
  if ! is_known_tool "$tool"; then
    echo "Unknown tool: ${tool}" >&2
    exit 1
  fi

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

  if [[ -n "${job_overrides[$tool]:-}" ]]; then
    jobs="${job_overrides[$tool]}"
  elif (( jobs_default_set )); then
    jobs="$jobs_default"
  else
    jobs="${recommended_jobs[$tool]}"
  fi
  require_positive_int "$jobs" "job count for ${tool}"
  echo "${tool}: ${#configs[@]} chunks, ${jobs} parallel worker(s)"

  if (( jobs == 1 || ${#configs[@]} == 1 )); then
    for config in "${configs[@]}"; do
      run_chunk "$tool" "$prefix" "$config"
    done
    continue
  fi

  printf '%s\0' "${configs[@]}" | xargs -0 -P "$jobs" -I{} bash -lc '
    set -euo pipefail
    config="$1"
    tool="$2"
    prefix="$3"
    chunk_id="$(basename "${config%.yaml}")"
    python tools/inference.py --tool "$tool" --output-name "${prefix}/${chunk_id}" --config "$config"
  ' _ "{}" "$tool" "$prefix"
done
