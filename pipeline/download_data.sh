#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-data/raw/diverse20}"
VALIDATOR="${ROOT_DIR}/pipeline/dataset/validate_bed_genome.py"
NORMALIZER="${ROOT_DIR}/pipeline/dataset/normalize_bed_chroms.py"

SPECIES_CODES=(
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

species_build_and_url() {
    case "$1" in
        hsa) echo "hg38 https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz" ;;
        mmu) echo "mm39 https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.fa.gz" ;;
        cfa) echo "canFam3 https://hgdownload.soe.ucsc.edu/goldenPath/canFam3/bigZips/canFam3.fa.gz" ;;
        cpo) echo "cavPor3 https://hgdownload.soe.ucsc.edu/goldenPath/cavPor3/bigZips/cavPor3.fa.gz" ;;
        ocu) echo "oryCun2 https://hgdownload.soe.ucsc.edu/goldenPath/oryCun2/bigZips/oryCun2.fa.gz" ;;
        eca) echo "equCab3 https://hgdownload.soe.ucsc.edu/goldenPath/equCab3/bigZips/equCab3.fa.gz" ;;
        mdo) echo "monDom5 https://hgdownload.soe.ucsc.edu/goldenPath/monDom5/bigZips/monDom5.fa.gz" ;;
        bta) echo "bosTau9 https://hgdownload.soe.ucsc.edu/goldenPath/bosTau9/bigZips/bosTau9.fa.gz" ;;
        gga) echo "galGal6 https://hgdownload.soe.ucsc.edu/goldenPath/galGal6/bigZips/galGal6.fa.gz" ;;
        tgu) echo "taeGut2 https://hgdownload.soe.ucsc.edu/goldenPath/taeGut2/bigZips/taeGut2.fa.gz" ;;
        aca) echo "anoCar2 https://hgdownload.soe.ucsc.edu/goldenPath/anoCar2/bigZips/anoCar2.fa.gz" ;;
        cpi) echo "chrPic1 https://hgdownload.soe.ucsc.edu/goldenPath/chrPic1/bigZips/chrPic1.fa.gz" ;;
        xtr) echo "xenTro10 https://hgdownload.soe.ucsc.edu/goldenPath/xenTro10/bigZips/xenTro10.fa.gz" ;;
        lch) echo "latCha1 https://hgdownload.soe.ucsc.edu/goldenPath/latCha1/bigZips/latCha1.fa.gz" ;;
        dre) echo "danRer11 https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/bigZips/danRer11.fa.gz" ;;
        cmi) echo "calMil1 https://hgdownload.soe.ucsc.edu/goldenPath/calMil1/bigZips/calMil1.fa.gz" ;;
        tni) echo "tetNig2 https://hgdownload.soe.ucsc.edu/goldenPath/tetNig2/bigZips/tetNig2.fa.gz" ;;
        cin) echo "ci3 https://hgdownload.soe.ucsc.edu/goldenPath/ci3/bigZips/ci3.fa.gz" ;;
        dme) echo "dm6 https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz" ;;
        cel) echo "ce11 https://hgdownload.soe.ucsc.edu/goldenPath/ce11/bigZips/ce11.fa.gz" ;;
        *)
            echo "Unsupported species code: $1"
            return 2
            ;;
    esac
}

report_value() {
    local report="$1"
    local key="$2"
    awk -F': ' -v key="$key" '$1 == key {print $2; exit}' "$report" 2>/dev/null || true
}

print_species_line() {
    local code="$1"
    local status="$2"
    local bed_rows="${3:-}"
    local matched_rows="${4:-}"
    local dropped_rows="${5:-}"
    local note="${6:-}"

    if [ -n "$bed_rows$matched_rows$dropped_rows" ]; then
        printf "%-4s %-42s %-20s bed=%s matched=%s dropped=%s\n" \
            "$code" "$(species_name "$code")" "$status" "$bed_rows" "$matched_rows" "$dropped_rows"
    elif [ -n "$note" ]; then
        printf "%-4s %-42s %-20s %s\n" "$code" "$(species_name "$code")" "$status" "$note"
    else
        printf "%-4s %-42s %-20s\n" "$code" "$(species_name "$code")" "$status"
    fi
}

ensure_genome() {
    local build="$1"
    local genome_url="$2"
    local out_dir="$3"
    local fasta="${out_dir}/${build}.fa"
    local gzip_path="${fasta}.gz"
    local temp_path="${gzip_path}.download"

    if [ -s "$fasta" ]; then
        return 0
    fi

    if [ -s "$gzip_path" ]; then
        if gunzip -f "$gzip_path" >/dev/null 2>&1 && [ -s "$fasta" ]; then
            return 0
        fi
        rm -f "$gzip_path" "$fasta"
    fi

    rm -f "$temp_path"
    if ! curl -fsSL -o "$temp_path" "$genome_url"; then
        rm -f "$temp_path"
        echo "genome_download_failed"
        return 1
    fi
    mv "$temp_path" "$gzip_path"
    if ! gunzip -f "$gzip_path" >/dev/null 2>&1 || [ ! -s "$fasta" ]; then
        rm -f "$gzip_path" "$fasta"
        echo "genome_decompression_failed"
        return 1
    fi
}

ensure_precursor_bed() {
    local code="$1"
    local out_dir="$2"
    local all_bed="${out_dir}/${code}-all.bed"
    local temp_bed="${all_bed}.download"
    local pre_bed="${out_dir}/${code}-precursors-no-v2.bed"
    local temp_pre="${pre_bed}.tmp"

    if [ -s "$pre_bed" ]; then
        return 0
    fi

    rm -f "$temp_bed" "$temp_pre"
    if ! curl -fsSL -o "$temp_bed" "https://mirgenedb.org/static/data/${code}/${code}-all.bed"; then
        rm -f "$temp_bed"
        echo "mirgenedb_bed_download_failed"
        return 1
    fi
    if ! awk '$4 ~ /_pre$/ && $0 !~ /-v2_/' "$temp_bed" > "$temp_pre" || [ ! -s "$temp_pre" ]; then
        rm -f "$temp_bed" "$temp_pre"
        echo "no_precursor_bed_rows"
        return 1
    fi
    mv "$temp_pre" "$pre_bed"
    rm -f "$temp_bed" "$all_bed"
}

ensure_alias() {
    local build="$1"
    local out_dir="$2"
    local alias_url="https://hgdownload.soe.ucsc.edu/goldenPath/${build}/database/chromAlias.txt.gz"
    local alias_path="${out_dir}/chromAlias.txt"
    local temp_alias="${alias_path}.gz.download"

    if [ -s "$alias_path" ]; then
        return 0
    fi

    rm -f "$temp_alias"
    if curl -fsSL -o "$temp_alias" "$alias_url"; then
        mv "$temp_alias" "${alias_path}.gz"
        gunzip -f "${alias_path}.gz" >/dev/null 2>&1 || rm -f "${alias_path}.gz" "$alias_path"
    else
        rm -f "$temp_alias"
    fi
}

download_species_data() {
    local code="$1"
    local out_dir="$2"
    local build="$3"
    local genome_url="$4"

    mkdir -p "$out_dir"
    ensure_genome "$build" "$genome_url" "$out_dir" || return 1
    ensure_precursor_bed "$code" "$out_dir" || return 1
    ensure_alias "$build" "$out_dir"
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
        print_species_line "$code" "auto_reused" "$(printf "%s\n" "$row" | awk -F'\t' '{print $7}')" "$(printf "%s\n" "$row" | awk -F'\t' '{print $8}')" "$(printf "%s\n" "$row" | awk -F'\t' '{print $9}')"
        return 0
    fi
    return 1
}

echo "Downloading final species panel to ${OUT_ROOT}"
success_count=0
failed_count=0
for code in "${SPECIES_CODES[@]}"; do
    out_dir="${OUT_ROOT}/${code}"
    if reuse_panel_row "$code"; then
        success_count=$((success_count + 1))
        continue
    fi

    if ! build_and_url="$(species_build_and_url "$code")"; then
        mkdir -p "$out_dir"
        validation="${out_dir}/download_error.txt"
        printf "%s\n" "$build_and_url" > "$validation"
        printf "%s\t%s\tdownload_failed\t\t\t%s\t\t\t\n" "$code" "$(species_name "$code")" "$validation" >> "$PANEL_TSV"
        print_species_line "$code" "download_failed" "" "" "" "$build_and_url"
        failed_count=$((failed_count + 1))
        continue
    fi
    read -r build genome_url <<< "$build_and_url"

    if ! download_reason="$(download_species_data "$code" "$out_dir" "$build" "$genome_url")"; then
        validation="${out_dir}/download_error.txt"
        printf "%s\n" "$download_reason" > "$validation"
        printf "%s\t%s\tdownload_failed\t\t\t%s\t\t\t\n" "$code" "$(species_name "$code")" "$validation" >> "$PANEL_TSV"
        print_species_line "$code" "download_failed" "" "" "" "$download_reason"
        failed_count=$((failed_count + 1))
        continue
    fi

    genome="${out_dir}/${build}.fa"
    raw_bed="${out_dir}/${code}-precursors-no-v2.bed"
    bed="${out_dir}/${code}-precursors-no-v2.normalized.bed"
    alias="${out_dir}/chromAlias.txt"
    normalize_report="${out_dir}/bed_chrom_normalization.txt"
    validation="${out_dir}/bed_genome_validation.txt"

    if [ ! -s "$genome" ] || [ ! -s "$raw_bed" ]; then
        printf "missing required downloaded files\n" > "$validation"
        printf "%s\t%s\tdownload_failed\t%s\t%s\t%s\t\t\t\n" "$code" "$(species_name "$code")" "$genome" "$raw_bed" "$validation" >> "$PANEL_TSV"
        print_species_line "$code" "download_failed" "" "" "" "missing required downloaded files"
        failed_count=$((failed_count + 1))
        continue
    fi

    normalize_cmd=(python "$NORMALIZER" --bed "$raw_bed" --genome "$genome" --output "$bed" --report "$normalize_report")
    if [ -s "$alias" ]; then
        normalize_cmd+=(--alias "$alias")
    fi
    if ! "${normalize_cmd[@]}" >/dev/null 2>> "$normalize_report"; then
        bed_rows="$(report_value "$normalize_report" "bed_rows")"
        matched_rows="$(report_value "$normalize_report" "matched_rows")"
        dropped_rows="$(report_value "$normalize_report" "dropped_rows")"
        printf "%s\t%s\tnormalization_failed\t%s\t%s\t%s\t%s\t%s\t%s\n" "$code" "$(species_name "$code")" "$genome" "$raw_bed" "$normalize_report" "$bed_rows" "$matched_rows" "$dropped_rows" >> "$PANEL_TSV"
        print_species_line "$code" "normalization_failed" "$bed_rows" "$matched_rows" "$dropped_rows"
        failed_count=$((failed_count + 1))
        continue
    fi

    bed_rows="$(report_value "$normalize_report" "bed_rows")"
    matched_rows="$(report_value "$normalize_report" "matched_rows")"
    dropped_rows="$(report_value "$normalize_report" "dropped_rows")"
    if python "$VALIDATOR" --bed "$bed" --genome "$genome" > "$validation" 2>&1; then
        printf "%s\t%s\tauto\t%s\t%s\t%s\t%s\t%s\t%s\n" "$code" "$(species_name "$code")" "$genome" "$bed" "$validation" "$bed_rows" "$matched_rows" "$dropped_rows" >> "$PANEL_TSV"
        print_species_line "$code" "auto" "$bed_rows" "$matched_rows" "$dropped_rows"
        success_count=$((success_count + 1))
    else
        printf "%s\t%s\tvalidation_failed\t%s\t%s\t%s\t%s\t%s\t%s\n" "$code" "$(species_name "$code")" "$genome" "$bed" "$validation" "$bed_rows" "$matched_rows" "$dropped_rows" >> "$PANEL_TSV"
        print_species_line "$code" "validation_failed" "$bed_rows" "$matched_rows" "$dropped_rows"
        failed_count=$((failed_count + 1))
    fi
done

echo "panel manifest: $PANEL_TSV"
echo "auto species: $success_count"
echo "failed species: $failed_count"
