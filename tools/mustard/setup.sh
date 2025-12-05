#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source "$(conda info --base)/etc/profile.d/conda.sh"

# create conda env
if conda env list | grep -q "^mustard "; then
    echo "Conda env 'mustard' already exists, skipping..."
else
    conda env create -f environment.yml
fi

git clone https://gitlab.com/RBP_Bioinformatics/mustard.git mustard_src

mkdir -p data/models data/test_loci data/ground_truth

# get models
for model in MuStARD-mirS MuStARD-mirF MuStARD-mirSF MuStARD-mirSC MuStARD-mirFC MuStARD-mirSFC MuStARD-mirSFC-U; do
    mkdir -p data/models/$model
    wget -O data/models/$model/CNNonRaw.hdf5 "https://gitlab.com/RBP_Bioinformatics/mustard_paper/-/raw/master/pretrained_models/$model/CNNonRaw.hdf5"
done

# get test data
wget -O data/test_loci/hsa.hairpin.slop5k.bothStrands.chr14.bed "https://gitlab.com/RBP_Bioinformatics/mustard_paper/-/raw/master/test_loci/miRNA/hsa/hsa.hairpin.slop5k.bothStrands.chr14.bed"
wget -O data/ground_truth/hsa.hairpin.chr14.bed "https://gitlab.com/RBP_Bioinformatics/mustard_paper/-/raw/master/annotation_files/miRNA/hsa/hsa.hairpin.chr14.bed"

wget -O data/chr14.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr14.fa.gz && gunzip -f data/chr14.fa.gz
wget -O data/chr14.wigFix.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.100way.phyloP100way/chr14.phyloP100way.wigFix.gz

echo "Setup complete!"
