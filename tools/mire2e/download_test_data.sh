#!/bin/bash
set -e

cd "$(dirname "$0")"

mkdir -p data/examples

wget -O data/examples/chr19_13836201_13836660_true.fa "https://raw.githubusercontent.com/sinc-lab/miRe2e/master/examples/chr19_13836201_13836660_true.fa"
wget -O data/examples/chr19_12003600_12004200_false.fa "https://raw.githubusercontent.com/sinc-lab/miRe2e/master/examples/chr19_12003600_12004200_false.fa"
