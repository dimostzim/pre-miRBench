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
if [ "${CHR}" != "all" ]; then
    grep "^${CHR}" "${OUT_DIR}/hg38.gtf" > "${OUT_DIR}/hg38-${CHR}.gtf"
    mv "${OUT_DIR}/hg38-${CHR}.gtf" "${OUT_DIR}/hg38.gtf"
fi

# Download repeat-masked chromosome
echo "downloading repeat-masked ${CHR}..."
curl -sL -o "${OUT_DIR}/${CHR}.fa.masked.gz" "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/${CHR}.fa.masked.gz"
gunzip -f "${OUT_DIR}/${CHR}.fa.masked.gz"
mv "${OUT_DIR}/${CHR}.fa.masked" "${OUT_DIR}/${CHR}.masked.fa"

echo ""
echo "downloaded:"
echo "  - genome: ${OUT_DIR}/${CHR}.fa"
echo "  - repeat-masked genome: ${OUT_DIR}/${CHR}.masked.fa"
echo "  - GTF annotation: ${OUT_DIR}/hg38.gtf"
echo "  - miRNA precursors: $(wc -l < ${OUT_DIR}/hsa-precursors-no-v2.bed) entries"
