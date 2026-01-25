#!/bin/bash
set -e

OUT_DIR="${1:-data}"
mkdir -p "${OUT_DIR}"

# Download C. elegans genome (ce11/WBcel235)
echo "downloading genome..."
curl -sL -o "${OUT_DIR}/ce11.fa.gz" "https://hgdownload.soe.ucsc.edu/goldenPath/ce11/bigZips/ce11.fa.gz"
gunzip -f "${OUT_DIR}/ce11.fa.gz"

# Download miRNA precursor coordinates from MirGeneDB
echo "downloading miRNA precursors..."
curl -sL -o "${OUT_DIR}/cel-all.bed" "https://mirgenedb.org/static/data/cel/cel-all.bed"
awk '$4 ~ /_pre$/' "${OUT_DIR}/cel-all.bed" | grep -v "\-v2_" > "${OUT_DIR}/cel-precursors-no-v2.bed"
rm "${OUT_DIR}/cel-all.bed"

# Download GTF annotation from Ensembl (for genomic region filtering)
echo "downloading GTF annotation..."
curl -sL -o "${OUT_DIR}/ce11.gtf.gz" "https://ftp.ensembl.org/pub/release-113/gtf/caenorhabditis_elegans/Caenorhabditis_elegans.WBcel235.113.gtf.gz"
gunzip -f "${OUT_DIR}/ce11.gtf.gz"

# Download repeat-masked genome (repeats masked with N)
echo "downloading repeat-masked genome..."
curl -sL -o "${OUT_DIR}/chromFaMasked.tar.gz" "https://hgdownload.soe.ucsc.edu/goldenPath/ce11/bigZips/chromFaMasked.tar.gz"
tar -xzf "${OUT_DIR}/chromFaMasked.tar.gz" -C "${OUT_DIR}"
cat "${OUT_DIR}"/*.fa.masked > "${OUT_DIR}/ce11.masked.fa"
rm "${OUT_DIR}"/*.fa.masked "${OUT_DIR}/chromFaMasked.tar.gz"

echo ""
echo "downloaded:"
echo "  - genome: ${OUT_DIR}/ce11.fa"
echo "  - repeat-masked genome: ${OUT_DIR}/ce11.masked.fa"
echo "  - GTF annotation: ${OUT_DIR}/ce11.gtf"
echo "  - miRNA precursors: $(wc -l < ${OUT_DIR}/cel-precursors-no-v2.bed) entries"
