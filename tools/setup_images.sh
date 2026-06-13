#!/bin/bash
set -euo pipefail

TOOLS=(deepmir deepmirgene dnnpremir mirdnn mire2e mustard)
REQUESTED_TOOL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tool)
            REQUESTED_TOOL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

build_image() {
    local tool="$1"
    local image_tag="${IMAGE_TAG:-${tool}:latest}"
    local build_args=()

    [ -n "${http_proxy:-}" ] && build_args+=(--build-arg "http_proxy=${http_proxy}")
    [ -n "${https_proxy:-}" ] && build_args+=(--build-arg "https_proxy=${https_proxy}")
    [ -n "${HTTP_PROXY:-}" ] && build_args+=(--build-arg "HTTP_PROXY=${HTTP_PROXY}")
    [ -n "${HTTPS_PROXY:-}" ] && build_args+=(--build-arg "HTTPS_PROXY=${HTTPS_PROXY}")

    docker build "${build_args[@]}" -t "$image_tag" "$(dirname "$0")/$tool"
}

if [ -n "$REQUESTED_TOOL" ]; then
    build_image "$REQUESTED_TOOL"
else
    for tool in "${TOOLS[@]}"; do
        build_image "$tool"
    done
fi
