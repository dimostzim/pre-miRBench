#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "$1" = "--docker" ]; then
    image_tag="${IMAGE_TAG:-mustard:latest}"
    docker build -t "$image_tag" .
    exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create -f environment.yml

git clone https://gitlab.com/RBP_Bioinformatics/mustard.git mustard_src

mkdir -p data/models

for model in MuStARD-mirS MuStARD-mirF MuStARD-mirSF MuStARD-mirSC MuStARD-mirFC MuStARD-mirSFC MuStARD-mirSFC-U; do
    mkdir -p data/models/$model
    wget -O data/models/$model/CNNonRaw.hdf5 "https://gitlab.com/RBP_Bioinformatics/mustard_paper/-/raw/master/pretrained_models/$model/CNNonRaw.hdf5"
done