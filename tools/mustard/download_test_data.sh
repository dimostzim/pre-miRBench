#!/bin/bash
set -e

cd "$(dirname "$0")"

mkdir -p data/test_loci data/ground_truth

wget -O data/test_loci/hsa.hairpin.slop5k.bothStrands.chr14.bed "https://gitlab.com/RBP_Bioinformatics/mustard_paper/-/raw/master/test_loci/miRNA/hsa/hsa.hairpin.slop5k.bothStrands.chr14.bed"
wget -O data/ground_truth/hsa.hairpin.chr14.bed "https://gitlab.com/RBP_Bioinformatics/mustard_paper/-/raw/master/annotation_files/miRNA/hsa/hsa.hairpin.chr14.bed"

wget -O data/chr14.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr14.fa.gz
gunzip -f data/chr14.fa.gz

wget -O data/chr14.wigFix.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.100way.phyloP100way/chr14.phyloP100way.wigFix.gz
