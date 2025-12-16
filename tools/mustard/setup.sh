#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$*" == *"--docker"* ]]; then
    image_tag="${IMAGE_TAG:-mustard:latest}"

    BUILD_ARGS=""
    [ -n "$http_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg http_proxy=$http_proxy"
    [ -n "$https_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg https_proxy=$https_proxy"
    [ -n "$HTTP_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTP_PROXY=$HTTP_PROXY"
    [ -n "$HTTPS_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTPS_PROXY=$HTTPS_PROXY"

    docker build --platform linux/amd64 $BUILD_ARGS -t "$image_tag" .
    exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create -f environment.yml

if [ ! -d "mustard_src" ]; then
    git clone https://gitlab.com/RBP_Bioinformatics/mustard.git mustard_src
fi

mkdir -p data/models

for model in MuStARD-mirS MuStARD-mirF MuStARD-mirSF MuStARD-mirSC MuStARD-mirFC MuStARD-mirSFC MuStARD-mirSFC-U; do
    mkdir -p data/models/$model
    wget -O data/models/$model/CNNonRaw.hdf5 "https://gitlab.com/RBP_Bioinformatics/mustard_paper/-/raw/master/pretrained_models/$model/CNNonRaw.hdf5"
done
