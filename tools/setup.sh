#!/bin/bash
set -e

TOOL=""
DOCKER_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --tool)
            TOOL="$2"
            shift 2
            ;;
        --docker)
            DOCKER_FLAG="--docker"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$TOOL" ]; then
    echo "Error: --tool is required"
    exit 1
fi

cd "$(dirname "$0")/$TOOL"
./setup.sh $DOCKER_FLAG
