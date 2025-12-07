#!/bin/bash
set -e

TOOL="$2"
DOCKER_FLAG="$3"

cd "$(dirname "$0")/tools/$TOOL"
./setup.sh $DOCKER_FLAG
