#!/bin/bash
set -e

OUT_DIR="${1:-data}"
GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/ce11/bigZips/ce11.fa.gz"

mkdir -p "${OUT_DIR}"
curl -L -o "${OUT_DIR}/ce11.fa.gz" "${GENOME_URL}"
gunzip -f "${OUT_DIR}/ce11.fa.gz"

echo "Downloaded C. elegans genome to ${OUT_DIR}/ce11.fa"
