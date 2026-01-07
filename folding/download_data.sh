#!/bin/bash
set -e

OUT_DIR="${1:-data}"
mkdir -p "${OUT_DIR}"

curl -sL -o "${OUT_DIR}/ce11.fa.gz" "https://hgdownload.soe.ucsc.edu/goldenPath/ce11/bigZips/ce11.fa.gz"
gunzip -f "${OUT_DIR}/ce11.fa.gz"

curl -sL -o "${OUT_DIR}/cel-all.bed" "https://mirgenedb.org/static/data/cel/cel-all.bed"
awk '$4 ~ /_pre$/' "${OUT_DIR}/cel-all.bed" | grep -v "\-v2_" > "${OUT_DIR}/cel-precursors-no-v2.bed"
rm "${OUT_DIR}/cel-all.bed"

echo "downloaded and prepared $(wc -l < ${OUT_DIR}/cel-precursors-no-v2.bed) precursors"
