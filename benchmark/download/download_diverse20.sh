#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${1:-data/train/raw/diverse20}"
VALIDATOR="${ROOT_DIR}/benchmark/train_data/validate_bed_genome.py"

AUTO_CODES=(
  hsa mmu mdo oan bta gga ami aca cpi xtr dre cmi gmo tni bfl cin dme aga cel spu
)

species_name() {
    case "$1" in
        hsa) echo "Human (Homo sapiens)" ;;
        mmu) echo "Mouse (Mus musculus)" ;;
        mdo) echo "Opossum (Monodelphis domestica)" ;;
        oan) echo "Platypus (Ornithorhynchus anatinus)" ;;
        bta) echo "Cow (Bos taurus)" ;;
        gga) echo "Chicken (Gallus gallus)" ;;
        ami) echo "Alligator (Alligator mississippiensis)" ;;
        aca) echo "Anole lizard (Anolis carolinensis)" ;;
        cpi) echo "Painted turtle (Chrysemys picta bellii)" ;;
        xtr) echo "Xenopus tropicalis" ;;
        dre) echo "Zebrafish (Danio rerio)" ;;
        cmi) echo "Elephant shark (Callorhinchus milii)" ;;
        gmo) echo "Atlantic cod (Gadus morhua)" ;;
        tni) echo "Tetraodon (Tetraodon nigroviridis)" ;;
        bfl) echo "Amphioxus (Branchiostoma floridae)" ;;
        cin) echo "Ciona intestinalis" ;;
        dme) echo "Drosophila melanogaster" ;;
        aga) echo "Anopheles gambiae" ;;
        cel) echo "C. elegans (Caenorhabditis elegans)" ;;
        spu) echo "Sea urchin (Strongylocentrotus purpuratus)" ;;
        *) echo "$1" ;;
    esac
}

mkdir -p "$OUT_ROOT"
PANEL_TSV="${OUT_ROOT}/panel.tsv"
printf "code\tspecies\tstatus\tgenome\tbed\tvalidation\n" > "$PANEL_TSV"

echo "Downloading auto-supported diverse panel species..."
for code in "${AUTO_CODES[@]}"; do
    out_dir="${OUT_ROOT}/${code}"
    echo "### ${code} $(species_name "$code")"
    "${ROOT_DIR}/benchmark/download/download_species.sh" "$code" "$out_dir" all

    genome="$(find "$out_dir" -maxdepth 1 -name "*.fa" | sort | head -n 1)"
    bed="${out_dir}/${code}-precursors-no-v2.bed"
    validation="${out_dir}/bed_genome_validation.txt"
    python "$VALIDATOR" --bed "$bed" --genome "$genome" > "$validation"
    printf "%s\t%s\tauto\t%s\t%s\t%s\n" "$code" "$(species_name "$code")" "$genome" "$bed" "$validation" >> "$PANEL_TSV"
done

echo ""
echo "panel manifest: $PANEL_TSV"
echo "all panel species include automatic UCSC genome download and BED-vs-genome validation"
