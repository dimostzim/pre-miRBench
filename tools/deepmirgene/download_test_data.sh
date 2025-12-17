#!/bin/bash
set -e

cd "$(dirname "$0")"

mkdir -p data

wget -O data/examples.fa "https://raw.githubusercontent.com/eleventh83/deepMiRGene/master/inference/examples.fa"
