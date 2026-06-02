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
  mdo  opossum      monDom5
  oan  platypus     ornAna2
  gga  chicken      galGal6
  aca  anole        anoCar2
  xtr  X. tropicalis xenTro10
  cmi  elephant shark calMil1
  bfl  amphioxus    braFlo1
  cin  Ciona        ci3
  dme  fruit fly    dm6
  dre  zebrafish    danRer11
  aga  A. gambiae   anoGam3
  cel  C. elegans   ce11
  spu  sea urchin   strPur2

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
    mdo) BUILD="monDom5"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/monDom5/bigZips/monDom5.fa.gz" ;;
    oan) BUILD="ornAna2"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/ornAna2/bigZips/ornAna2.fa.gz" ;;
    gga) BUILD="galGal6"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/galGal6/bigZips/galGal6.fa.gz" ;;
    aca) BUILD="anoCar2"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/anoCar2/bigZips/anoCar2.fa.gz" ;;
    xtr) BUILD="xenTro10"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/xenTro10/bigZips/xenTro10.fa.gz" ;;
    cmi) BUILD="calMil1"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/calMil1/bigZips/calMil1.fa.gz" ;;
    bfl) BUILD="braFlo1"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/braFlo1/bigZips/braFlo1.fa.gz" ;;
    cin) BUILD="ci3"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/ci3/bigZips/ci3.fa.gz" ;;
    dme) BUILD="dm6"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz" ;;
    dre) BUILD="danRer11"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/bigZips/danRer11.fa.gz" ;;
    aga) BUILD="anoGam3"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/anoGam3/bigZips/anoGam3.fa.gz" ;;
    cel) BUILD="ce11"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/ce11/bigZips/ce11.fa.gz" ;;
    spu) BUILD="strPur2"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/strPur2/bigZips/strPur2.fa.gz" ;;
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
