#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${1:-data/train/raw/diverse20}"
VALIDATOR="${ROOT_DIR}/benchmark/train_data/validate_bed_genome.py"
NORMALIZER="${ROOT_DIR}/benchmark/train_data/normalize_bed_chroms.py"

AUTO_CODES=(
  hsa mmu cfa cpo ocu eca mdo bta gga tgu aca cpi xtr lch dre cmi tni cin dme cel
)

species_name() {
    case "$1" in
        hsa) echo "Human (Homo sapiens)" ;;
        mmu) echo "Mouse (Mus musculus)" ;;
        cfa) echo "Dog (Canis familiaris)" ;;
        cpo) echo "Guinea pig (Cavia porcellus)" ;;
        ocu) echo "Rabbit (Oryctolagus cuniculus)" ;;
        eca) echo "Horse (Equus caballus)" ;;
        mdo) echo "Opossum (Monodelphis domestica)" ;;
        bta) echo "Cow (Bos taurus)" ;;
        gga) echo "Chicken (Gallus gallus)" ;;
        tgu) echo "Zebra finch (Taeniopygia guttata)" ;;
        aca) echo "Anole lizard (Anolis carolinensis)" ;;
        cpi) echo "Painted turtle (Chrysemys picta bellii)" ;;
        xtr) echo "Xenopus tropicalis" ;;
        lch) echo "Coelacanth (Latimeria chalumnae)" ;;
        dre) echo "Zebrafish (Danio rerio)" ;;
        cmi) echo "Elephant shark (Callorhinchus milii)" ;;
        tni) echo "Tetraodon (Tetraodon nigroviridis)" ;;
        cin) echo "Ciona intestinalis" ;;
        dme) echo "Drosophila melanogaster" ;;
        cel) echo "C. elegans (Caenorhabditis elegans)" ;;
        *) echo "$1" ;;
    esac
}

mkdir -p "$OUT_ROOT"
PANEL_TSV="${OUT_ROOT}/panel.tsv"
OLD_PANEL_TSV=""
if [ -s "$PANEL_TSV" ]; then
    OLD_PANEL_TSV="${PANEL_TSV}.previous"
    mv "$PANEL_TSV" "$OLD_PANEL_TSV"
fi
printf "code\tspecies\tstatus\tgenome\tbed\tvalidation\tbed_rows\tmatched_rows\tdropped_rows\n" > "$PANEL_TSV"

reuse_panel_row() {
    local code="$1"
    local row genome bed validation
    if [ -z "$OLD_PANEL_TSV" ]; then
        return 1
    fi
    row="$(awk -F'\t' -v code="$code" 'NR > 1 && $1 == code && $3 == "auto" {print; exit}' "$OLD_PANEL_TSV")"
    if [ -z "$row" ]; then
        return 1
    fi
    genome="$(printf "%s\n" "$row" | awk -F'\t' '{print $4}')"
    bed="$(printf "%s\n" "$row" | awk -F'\t' '{print $5}')"
    validation="$(printf "%s\n" "$row" | awk -F'\t' '{print $6}')"
    if [ -s "$genome" ] && [ -s "$bed" ] && [ -s "$validation" ]; then
        printf "%s\n" "$row" >> "$PANEL_TSV"
        return 0
    fi
    return 1
}

echo "Downloading auto-supported diverse panel species..."
success_count=0
failed_count=0
for code in "${AUTO_CODES[@]}"; do
    out_dir="${OUT_ROOT}/${code}"
    echo "### ${code} $(species_name "$code")"
    if reuse_panel_row "$code"; then
        success_count=$((success_count + 1))
        echo "  status: auto (reused)"
        continue
    fi
    if ! "${ROOT_DIR}/benchmark/download/download_species.sh" "$code" "$out_dir" all; then
        validation="${out_dir}/download_error.txt"
        mkdir -p "$out_dir"
        echo "download failed" > "$validation"
        printf "%s\t%s\tdownload_failed\t\t\t%s\t\t\t\n" "$code" "$(species_name "$code")" "$validation" >> "$PANEL_TSV"
        failed_count=$((failed_count + 1))
        echo "  status: download_failed"
        continue
    fi

    genome="$(find "$out_dir" -maxdepth 1 -name "*.fa" | sort | head -n 1)"
    raw_bed="${out_dir}/${code}-precursors-no-v2.bed"
    bed="${out_dir}/${code}-precursors-no-v2.normalized.bed"
    alias="${out_dir}/chromAlias.txt"
    normalize_report="${out_dir}/bed_chrom_normalization.txt"
    validation="${out_dir}/bed_genome_validation.txt"
    normalize_cmd=(python "$NORMALIZER" --bed "$raw_bed" --genome "$genome" --output "$bed" --report "$normalize_report")
    if [ -s "$alias" ]; then
        normalize_cmd+=(--alias "$alias")
    fi
    if ! "${normalize_cmd[@]}"; then
        bed_rows="$(awk -F': ' '/^bed_rows:/ {print $2}' "$normalize_report" 2>/dev/null || true)"
        matched_rows="$(awk -F': ' '/^matched_rows:/ {print $2}' "$normalize_report" 2>/dev/null || true)"
        dropped_rows="$(awk -F': ' '/^dropped_rows:/ {print $2}' "$normalize_report" 2>/dev/null || true)"
        printf "%s\t%s\tnormalization_failed\t%s\t%s\t%s\t%s\t%s\t%s\n" "$code" "$(species_name "$code")" "$genome" "$raw_bed" "$normalize_report" "$bed_rows" "$matched_rows" "$dropped_rows" >> "$PANEL_TSV"
        failed_count=$((failed_count + 1))
        echo "  status: normalization_failed"
        echo "  normalization details: $normalize_report"
        continue
    fi
    bed_rows="$(awk -F': ' '/^bed_rows:/ {print $2}' "$normalize_report")"
    matched_rows="$(awk -F': ' '/^matched_rows:/ {print $2}' "$normalize_report")"
    dropped_rows="$(awk -F': ' '/^dropped_rows:/ {print $2}' "$normalize_report")"
    if python "$VALIDATOR" --bed "$bed" --genome "$genome" > "$validation"; then
        printf "%s\t%s\tauto\t%s\t%s\t%s\t%s\t%s\t%s\n" "$code" "$(species_name "$code")" "$genome" "$bed" "$validation" "$bed_rows" "$matched_rows" "$dropped_rows" >> "$PANEL_TSV"
        success_count=$((success_count + 1))
        echo "  status: auto"
    else
        printf "%s\t%s\tvalidation_failed\t%s\t%s\t%s\t%s\t%s\t%s\n" "$code" "$(species_name "$code")" "$genome" "$bed" "$validation" "$bed_rows" "$matched_rows" "$dropped_rows" >> "$PANEL_TSV"
        failed_count=$((failed_count + 1))
        echo "  status: validation_failed"
        echo "  validation details: $validation"
    fi
done

echo ""
echo "panel manifest: $PANEL_TSV"
echo "auto species: $success_count"
echo "failed species: $failed_count"
if [ "$failed_count" -gt 0 ]; then
    echo "failed rows are kept in panel.tsv but skipped by the dataset builder"
fi
