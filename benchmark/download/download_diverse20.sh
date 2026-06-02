#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${1:-data/train/raw/diverse20}"
VALIDATOR="${ROOT_DIR}/benchmark/train_data/validate_bed_genome.py"

AUTO_CODES=(
  hsa mmu mdo oan gga aca xtr dre cmi bfl cin dme aga cel spu
)

MANUAL_GENOME_CODES=(
  loc tca nve aqu sro
)

species_name() {
    case "$1" in
        hsa) echo "Human (Homo sapiens)" ;;
        mmu) echo "Mouse (Mus musculus)" ;;
        mdo) echo "Opossum (Monodelphis domestica)" ;;
        oan) echo "Platypus (Ornithorhynchus anatinus)" ;;
        gga) echo "Chicken (Gallus gallus)" ;;
        aca) echo "Anole lizard (Anolis carolinensis)" ;;
        xtr) echo "Xenopus tropicalis" ;;
        dre) echo "Zebrafish (Danio rerio)" ;;
        loc) echo "Spotted gar (Lepisosteus oculatus)" ;;
        cmi) echo "Elephant shark (Callorhinchus milii)" ;;
        bfl) echo "Amphioxus (Branchiostoma floridae)" ;;
        cin) echo "Ciona intestinalis" ;;
        dme) echo "Drosophila melanogaster" ;;
        tca) echo "Red flour beetle (Tribolium castaneum)" ;;
        aga) echo "Anopheles gambiae" ;;
        cel) echo "C. elegans (Caenorhabditis elegans)" ;;
        spu) echo "Sea urchin (Strongylocentrotus purpuratus)" ;;
        nve) echo "Sea anemone (Nematostella vectensis)" ;;
        aqu) echo "Sponge (Amphimedon queenslandica)" ;;
        sro) echo "Flatworm (Symsagittifera roscoffensis)" ;;
        *) echo "$1" ;;
    esac
}

download_bed_only() {
    local code="$1"
    local out_dir="$2"
    mkdir -p "$out_dir"
    local all_bed="${out_dir}/${code}-all.bed"
    local pre_bed="${out_dir}/${code}-precursors-no-v2.bed"
    curl -L -o "$all_bed" "https://mirgenedb.org/static/data/${code}/${code}-all.bed"
    awk '$4 ~ /_pre$/' "$all_bed" | grep -v -- "-v2_" > "$pre_bed"
    rm "$all_bed"
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

echo "Downloading MirGeneDB BEDs for manual-genome species..."
for code in "${MANUAL_GENOME_CODES[@]}"; do
    out_dir="${OUT_ROOT}/${code}"
    echo "### ${code} $(species_name "$code")"
    download_bed_only "$code" "$out_dir"
    bed="${out_dir}/${code}-precursors-no-v2.bed"
    printf "%s\t%s\tmanual_genome_required\t\t%s\t\n" "$code" "$(species_name "$code")" "$bed" >> "$PANEL_TSV"
done

echo ""
echo "panel manifest: $PANEL_TSV"
echo "manual genome required for: ${MANUAL_GENOME_CODES[*]}"
