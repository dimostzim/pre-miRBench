#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  benchmark/download/download_species.sh <species_code> [output_dir] [chromosome|all]

Supported species:
  hsa  human        hg38
  mmu  mouse        mm39
  rno  rat          rn7
  dme  fruit fly    dm6
  dre  zebrafish    danRer11
  cel  C. elegans   ce11

Outputs:
  <ucsc_build>.fa
  <species_code>-precursors-no-v2.bed

The optional chromosome filter keeps only BED rows whose first column matches
the supplied value. Use "all" to keep all chromosomes.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

SPECIES="${1:-}"
OUT_DIR="${2:-data/train/raw/${SPECIES}}"
CHROM="${3:-all}"

if [ -z "$SPECIES" ]; then
    usage >&2
    exit 2
fi

case "$SPECIES" in
    hsa) BUILD="hg38"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz" ;;
    mmu) BUILD="mm39"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.fa.gz" ;;
    rno) BUILD="rn7"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/rn7/bigZips/rn7.fa.gz" ;;
    dme) BUILD="dm6"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz" ;;
    dre) BUILD="danRer11"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/bigZips/danRer11.fa.gz" ;;
    cel) BUILD="ce11"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/ce11/bigZips/ce11.fa.gz" ;;
    *)
        echo "Unsupported species code: $SPECIES" >&2
        usage >&2
        exit 2
        ;;
esac

mkdir -p "$OUT_DIR"

echo "Downloading ${SPECIES} genome (${BUILD})..."
curl -L -o "${OUT_DIR}/${BUILD}.fa.gz" "$GENOME_URL"
gunzip -f "${OUT_DIR}/${BUILD}.fa.gz"

echo "Downloading ${SPECIES} MirGeneDB precursor coordinates..."
ALL_BED="${OUT_DIR}/${SPECIES}-all.bed"
PRE_BED="${OUT_DIR}/${SPECIES}-precursors-no-v2.bed"
curl -L -o "$ALL_BED" "https://mirgenedb.org/static/data/${SPECIES}/${SPECIES}-all.bed"
awk '$4 ~ /_pre$/' "$ALL_BED" | grep -v -- "-v2_" > "$PRE_BED"
rm "$ALL_BED"

if [ "$CHROM" != "all" ]; then
    FILTERED_BED="${OUT_DIR}/${SPECIES}-precursors-${CHROM}.bed"
    grep "^${CHROM}[[:space:]]" "$PRE_BED" > "$FILTERED_BED" || true
    mv "$FILTERED_BED" "$PRE_BED"
fi

echo ""
echo "downloaded:"
echo "  - genome: ${OUT_DIR}/${BUILD}.fa"
echo "  - precursors: ${PRE_BED} ($(wc -l < "$PRE_BED") entries)"
if [ "$SPECIES" != "hsa" ]; then
    echo "  - conservation: not downloaded; use MuStARD inputMode without conservation unless you provide tracks"
fi
