#!/bin/bash
set -e
cd "$(dirname "$0")"

mkdir -p data

wget -O data/test.fa "https://raw.githubusercontent.com/zhengxueming/dnnPreMiR/master/testData/hsa-pre-release22.fa"
