#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f environment.yml

git clone https://gitlab.com/RBP_Bioinformatics/mustard.git mustard_src
git clone https://gitlab.com/RBP_Bioinformatics/mustard_paper.git mustard_paper

mkdir -p data
wget -O data/chr14.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr14.fa.gz
gunzip -f data/chr14.fa.gz
wget -O data/chr14.phyloP100way.wigFix.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.100way.phyloP100way/chr14.phyloP100way.wigFix.gz
gunzip -f data/chr14.phyloP100way.wigFix.gz

echo "Setup complete!"
