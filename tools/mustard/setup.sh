#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

image_tag="${IMAGE_TAG:-mustard:latest}"

BUILD_ARGS=""
[ -n "$http_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg http_proxy=$http_proxy"
[ -n "$https_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg https_proxy=$https_proxy"
[ -n "$HTTP_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTP_PROXY=$HTTP_PROXY"
[ -n "$HTTPS_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTPS_PROXY=$HTTPS_PROXY"

docker build $BUILD_ARGS -t "$image_tag" .
