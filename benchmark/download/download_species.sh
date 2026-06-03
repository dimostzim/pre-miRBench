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
  cfa  dog          canFam3
  cpo  guinea pig   cavPor3
  ocu  rabbit       oryCun2
  eca  horse        equCab3
  mdo  opossum      monDom5
  bta  cow          bosTau9
  gga  chicken      galGal6
  tgu  zebra finch  taeGut2
  aca  anole        anoCar2
  cpi  painted turtle chrPic1
  xtr  X. tropicalis xenTro10
  xla  X. laevis    xenLae2
  lch  coelacanth   latCha1
  cmi  elephant shark calMil1
  tni  Tetraodon    tetNig2
  cin  Ciona        ci3
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
    cfa) BUILD="canFam3"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/canFam3/bigZips/canFam3.fa.gz" ;;
    cpo) BUILD="cavPor3"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/cavPor3/bigZips/cavPor3.fa.gz" ;;
    ocu) BUILD="oryCun2"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/oryCun2/bigZips/oryCun2.fa.gz" ;;
    eca) BUILD="equCab3"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/equCab3/bigZips/equCab3.fa.gz" ;;
    mdo) BUILD="monDom5"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/monDom5/bigZips/monDom5.fa.gz" ;;
    bta) BUILD="bosTau9"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/bosTau9/bigZips/bosTau9.fa.gz" ;;
    gga) BUILD="galGal6"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/galGal6/bigZips/galGal6.fa.gz" ;;
    tgu) BUILD="taeGut2"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/taeGut2/bigZips/taeGut2.fa.gz" ;;
    aca) BUILD="anoCar2"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/anoCar2/bigZips/anoCar2.fa.gz" ;;
    cpi) BUILD="chrPic1"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/chrPic1/bigZips/chrPic1.fa.gz" ;;
    xtr) BUILD="xenTro10"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/xenTro10/bigZips/xenTro10.fa.gz" ;;
    xla) BUILD="xenLae2"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/xenLae2/bigZips/xenLae2.fa.gz" ;;
    lch) BUILD="latCha1"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/latCha1/bigZips/latCha1.fa.gz" ;;
    cmi) BUILD="calMil1"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/calMil1/bigZips/calMil1.fa.gz" ;;
    tni) BUILD="tetNig2"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/tetNig2/bigZips/tetNig2.fa.gz" ;;
    cin) BUILD="ci3"; GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/ci3/bigZips/ci3.fa.gz" ;;
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

if [ -s "${OUT_DIR}/${BUILD}.fa" ]; then
    echo "Using existing ${SPECIES} genome (${BUILD}): ${OUT_DIR}/${BUILD}.fa"
else
    echo "Downloading ${SPECIES} genome (${BUILD})..."
    curl -L -o "${OUT_DIR}/${BUILD}.fa.gz" "$GENOME_URL"
    gunzip -f "${OUT_DIR}/${BUILD}.fa.gz"
fi

echo "Downloading ${SPECIES} MirGeneDB precursor coordinates..."
ALL_BED="${OUT_DIR}/${SPECIES}-all.bed"
PRE_BED="${OUT_DIR}/${SPECIES}-precursors-no-v2.bed"
if [ -s "$PRE_BED" ] && [ "$CHROM" = "all" ]; then
    echo "Using existing ${SPECIES} MirGeneDB precursor coordinates: $PRE_BED"
else
    curl -L -o "$ALL_BED" "https://mirgenedb.org/static/data/${SPECIES}/${SPECIES}-all.bed"
    awk '$4 ~ /_pre$/' "$ALL_BED" | grep -v -- "-v2_" > "$PRE_BED"
    rm "$ALL_BED"
fi

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

ALIAS_URL="https://hgdownload.soe.ucsc.edu/goldenPath/${BUILD}/database/chromAlias.txt.gz"
ALIAS_PATH="${OUT_DIR}/chromAlias.txt"
if [ ! -s "$ALIAS_PATH" ]; then
    if curl -fsSL -o "${ALIAS_PATH}.gz" "$ALIAS_URL"; then
        gunzip -f "${ALIAS_PATH}.gz"
        echo "  - aliases: ${ALIAS_PATH}"
    else
        rm -f "${ALIAS_PATH}.gz"
        echo "  - aliases: not available"
    fi
else
    echo "  - aliases: ${ALIAS_PATH}"
fi
