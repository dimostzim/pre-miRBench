#!/bin/bash
set -e

CHR="${1:-chr14}"
OUT_DIR="${2:-data}"
mkdir -p "${OUT_DIR}"

echo "Downloading human ${CHR} data..."

# Download human chromosome
echo "downloading ${CHR}..."
curl -sL -o "${OUT_DIR}/${CHR}.fa.gz" "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/${CHR}.fa.gz"
gunzip -f "${OUT_DIR}/${CHR}.fa.gz"

# Download chromosome-specific phyloP conservation track for MuStARD.
if [ "${CHR}" != "all" ]; then
    echo "downloading ${CHR} conservation..."
    curl -sL -o "${OUT_DIR}/${CHR}.wigFix.gz" "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.100way.phyloP100way/${CHR}.phyloP100way.wigFix.gz"
fi

# Download miRNA precursor coordinates from MirGeneDB
echo "downloading miRNA precursors..."
curl -sL -o "${OUT_DIR}/hsa-all.bed" "https://mirgenedb.org/static/data/hsa/hsa-all.bed"
awk '$4 ~ /_pre$/' "${OUT_DIR}/hsa-all.bed" | grep -v "\-v2_" > "${OUT_DIR}/hsa-precursors-no-v2.bed"
# Filter to specific chromosome if not all
if [ "${CHR}" != "all" ]; then
    grep "^${CHR}" "${OUT_DIR}/hsa-precursors-no-v2.bed" > "${OUT_DIR}/hsa-precursors-${CHR}.bed"
    mv "${OUT_DIR}/hsa-precursors-${CHR}.bed" "${OUT_DIR}/hsa-precursors-no-v2.bed"
fi
rm "${OUT_DIR}/hsa-all.bed"

# Download GTF annotation from Ensembl (for genomic region filtering)
echo "downloading GTF annotation..."
curl -sL -o "${OUT_DIR}/hg38.gtf.gz" "https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/Homo_sapiens.GRCh38.113.gtf.gz"
gunzip -f "${OUT_DIR}/hg38.gtf.gz"
# Filter to specific chromosome if not all
# Ensembl GTF uses bare numbers (e.g. "14"), not "chr14"
if [ "${CHR}" != "all" ]; then
    ENS_CHR="${CHR#chr}"
    grep "^${ENS_CHR}" "${OUT_DIR}/hg38.gtf" > "${OUT_DIR}/hg38-${CHR}.gtf" || true
    mv "${OUT_DIR}/hg38-${CHR}.gtf" "${OUT_DIR}/hg38.gtf"
fi

echo ""
echo "downloaded:"
echo "  - genome: ${OUT_DIR}/${CHR}.fa (soft-masked: repeats in lowercase)"
if [ "${CHR}" != "all" ]; then
    echo "  - conservation: ${OUT_DIR}/${CHR}.wigFix.gz"
else
    echo "  - conservation: skipped for CHR=all"
fi
echo "  - GTF annotation: ${OUT_DIR}/hg38.gtf"
echo "  - miRNA precursors: $(wc -l < ${OUT_DIR}/hsa-precursors-no-v2.bed) entries"
