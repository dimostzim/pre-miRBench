#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-data/raw/mirgenedb_71}"
PANEL_SOURCE="${2:-${ROOT_DIR}/panels/mirgenedb_71/species_panel.tsv}"
VALIDATOR="${ROOT_DIR}/pipeline/dataset/validate_bed_genome.py"
NORMALIZER="${ROOT_DIR}/pipeline/dataset/normalize_bed_chroms.py"
SPECIES_FILTER="${SPECIES:-}"

resolve_path() {
    local value="$1"
    if [ -z "$value" ]; then
        return 0
    fi
    if [[ "$value" = /* ]]; then
        printf "%s\n" "$value"
    else
        printf "%s\n" "${ROOT_DIR}/${value}"
    fi
}

report_value() {
    local report="$1"
    local key="$2"
    awk -F': ' -v key="$key" '$1 == key {print $2; exit}' "$report" 2>/dev/null || true
}

should_download_species() {
    local code="$1"
    if [ -z "$SPECIES_FILTER" ]; then
        return 0
    fi
    IFS=',' read -ra requested <<< "$SPECIES_FILTER"
    for item in "${requested[@]}"; do
        if [ "$code" = "$item" ]; then
            return 0
        fi
    done
    return 1
}

print_species_line() {
    local code="$1"
    local species="$2"
    local status="$3"
    local bed_rows="${4:-}"
    local matched_rows="${5:-}"
    local dropped_rows="${6:-}"
    local note="${7:-}"

    if [ -n "$bed_rows$matched_rows$dropped_rows" ]; then
        printf "%-4s %-48s %-20s bed=%s matched=%s dropped=%s\n" \
            "$code" "$species" "$status" "$bed_rows" "$matched_rows" "$dropped_rows"
    elif [ -n "$note" ]; then
        printf "%-4s %-48s %-20s %s\n" "$code" "$species" "$status" "$note"
    else
        printf "%-4s %-48s %-20s\n" "$code" "$species" "$status"
    fi
}

materialize_fasta() {
    local download_path="$1"
    local fasta="$2"
    python - "$download_path" "$fasta" <<'PY'
import gzip
import shutil
import sys
import zipfile
from pathlib import Path

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
temp = dest.with_suffix(dest.suffix + ".tmp")


def valid_fasta(path):
    try:
        with path.open("rb") as handle:
            return handle.read(1) == b">"
    except OSError:
        return False


def copy_plain():
    shutil.copyfile(source, temp)


def copy_gzip():
    with gzip.open(source, "rb") as input_handle, temp.open("wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle)


def zip_candidates(archive):
    names = [
        name
        for name in archive.namelist()
        if not name.endswith("/")
        and name.lower().endswith((".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz"))
    ]

    def score(name):
        lower = name.lower()
        return (
            0 if "genomic" in lower else 1,
            0 if lower.endswith((".fna", ".fna.gz")) else 1,
            len(name),
            name,
        )

    return sorted(names, key=score)


def copy_zip():
    with zipfile.ZipFile(source) as archive:
        candidates = zip_candidates(archive)
        if not candidates:
            raise SystemExit(f"no FASTA file found inside {source}")
        name = candidates[0]
        with archive.open(name) as input_handle:
            if name.lower().endswith(".gz"):
                with gzip.GzipFile(fileobj=input_handle) as gzip_handle, temp.open("wb") as output_handle:
                    shutil.copyfileobj(gzip_handle, output_handle)
            else:
                with temp.open("wb") as output_handle:
                    shutil.copyfileobj(input_handle, output_handle)


if temp.exists():
    temp.unlink()

try:
    with source.open("rb") as handle:
        magic = handle.read(2)
    if magic == b"\x1f\x8b":
        copy_gzip()
    else:
        if zipfile.is_zipfile(source):
            copy_zip()
        else:
            copy_plain()
except Exception as exc:
    if temp.exists():
        temp.unlink()
    raise SystemExit(str(exc))

if not valid_fasta(temp):
    if temp.exists():
        temp.unlink()
    raise SystemExit(f"download did not contain a FASTA file: {source}")

temp.replace(dest)
PY
}

ensure_genome() {
    local genome_url="$1"
    local out_dir="$2"
    local fasta="${out_dir}/genome.fa"
    local download_path="${out_dir}/genome.download"

    if [ -s "$fasta" ]; then
        return 0
    fi

    rm -f "$download_path" "${fasta}.tmp"
    if ! curl --http1.1 --retry 5 --retry-all-errors --retry-delay 5 --connect-timeout 30 -fL -o "$download_path" "$genome_url"; then
        rm -f "$download_path"
        echo "genome_download_failed"
        return 1
    fi
    if ! materialize_fasta "$download_path" "$fasta"; then
        rm -f "$download_path" "$fasta" "${fasta}.tmp"
        echo "genome_materialization_failed"
        return 1
    fi
    rm -f "$download_path"
}

apply_supplement_fasta() {
    local fasta="$1"
    local supplement_fasta="$2"
    local first_header

    if [ -z "$supplement_fasta" ]; then
        return 0
    fi
    supplement_fasta="$(resolve_path "$supplement_fasta")"
    if [ ! -s "$supplement_fasta" ]; then
        echo "missing_supplement_fasta"
        return 1
    fi

    first_header="$(awk '/^>/ {print; exit}' "$supplement_fasta")"
    if [ -n "$first_header" ] && grep -Fqx "$first_header" "$fasta"; then
        return 0
    fi
    {
        printf "\n"
        cat "$supplement_fasta"
    } >> "$fasta"
}

ensure_precursor_bed() {
    local code="$1"
    local bed_url="$2"
    local out_dir="$3"
    local supplement_bed="$4"
    local all_bed="${out_dir}/${code}-all.bed"
    local temp_bed="${all_bed}.download"
    local pre_bed="${out_dir}/${code}-precursors.bed"
    local temp_pre="${pre_bed}.tmp"
    local temp_merged="${pre_bed}.merged"

    if [ -s "$pre_bed" ]; then
        return 0
    fi

    rm -f "$temp_bed" "$temp_pre" "$temp_merged"
    if ! curl --http1.1 --retry 5 --retry-all-errors --retry-delay 5 --connect-timeout 30 -fL -o "$temp_bed" "$bed_url"; then
        rm -f "$temp_bed"
        echo "mirgenedb_bed_download_failed"
        return 1
    fi
    mv "$temp_bed" "$all_bed"

    if ! awk '$4 ~ /_pre$/' "$all_bed" > "$temp_pre" || [ ! -s "$temp_pre" ]; then
        rm -f "$temp_pre" "$temp_merged"
        echo "no_precursor_bed_rows"
        return 1
    fi

    if [ -n "$supplement_bed" ]; then
        supplement_bed="$(resolve_path "$supplement_bed")"
        if [ ! -s "$supplement_bed" ]; then
            rm -f "$temp_pre" "$temp_merged"
            echo "missing_supplement_bed"
            return 1
        fi
        awk 'NR == FNR {skip[$4] = 1; next} !($4 in skip)' "$supplement_bed" "$temp_pre" > "$temp_merged"
        cat "$supplement_bed" >> "$temp_merged"
        mv "$temp_merged" "$temp_pre"
    fi

    mv "$temp_pre" "$pre_bed"
}

ensure_alias() {
    local alias_url="$1"
    local out_dir="$2"
    local alias_path="${out_dir}/chromAlias.txt"
    local temp_alias="${alias_path}.download"

    if [ -z "$alias_url" ] || [ -s "$alias_path" ]; then
        return 0
    fi

    rm -f "$temp_alias"
    if ! curl --http1.1 --retry 3 --retry-all-errors --retry-delay 3 --connect-timeout 30 -fL -o "$temp_alias" "$alias_url" 2>/dev/null; then
        rm -f "$temp_alias"
        return 0
    fi

    if gzip -t "$temp_alias" >/dev/null 2>&1; then
        gunzip -c "$temp_alias" > "$alias_path" || rm -f "$alias_path"
        rm -f "$temp_alias"
    else
        mv "$temp_alias" "$alias_path"
    fi
}

download_species_data() {
    local code="$1"
    local genome_url="$2"
    local bed_url="$3"
    local alias_url="$4"
    local out_dir="$5"
    local supplement_fasta="$6"
    local supplement_bed="$7"

    mkdir -p "$out_dir"
    ensure_genome "$genome_url" "$out_dir" || return 1
    apply_supplement_fasta "${out_dir}/genome.fa" "$supplement_fasta" || return 1
    ensure_precursor_bed "$code" "$bed_url" "$out_dir" "$supplement_bed" || return 1
    ensure_alias "$alias_url" "$out_dir"
}

if [ ! -s "$PANEL_SOURCE" ]; then
    echo "missing panel source: $PANEL_SOURCE" >&2
    exit 1
fi

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
    local species="$2"
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
        print_species_line "$code" "$species" "auto_reused" "$(printf "%s\n" "$row" | awk -F'\t' '{print $7}')" "$(printf "%s\n" "$row" | awk -F'\t' '{print $8}')" "$(printf "%s\n" "$row" | awk -F'\t' '{print $9}')"
        return 0
    fi
    return 1
}

echo "Downloading final species panel to ${OUT_ROOT}"
echo "panel source: ${PANEL_SOURCE}"
success_count=0
failed_count=0

while IFS=$'\034' read -r code common_name scientific_name taxonomy_group pre_mirna_count genome_source assembly_name assembly_accession genome_url alias_url bed_url supplement_fasta supplement_bed panel_status notes; do
    if ! should_download_species "$code"; then
        continue
    fi

    species="${common_name} (${scientific_name})"
    out_dir="${OUT_ROOT}/${code}"
    if reuse_panel_row "$code" "$species"; then
        success_count=$((success_count + 1))
        continue
    fi

    if [ -z "$genome_url" ] || [ -z "$bed_url" ]; then
        mkdir -p "$out_dir"
        validation="${out_dir}/download_error.txt"
        printf "missing genome or BED URL in panel source\n" > "$validation"
        printf "%s\t%s\tdownload_failed\t\t\t%s\t\t\t\n" "$code" "$species" "$validation" >> "$PANEL_TSV"
        print_species_line "$code" "$species" "download_failed" "" "" "" "missing genome or BED URL"
        failed_count=$((failed_count + 1))
        continue
    fi

    if ! download_reason="$(download_species_data "$code" "$genome_url" "$bed_url" "$alias_url" "$out_dir" "$supplement_fasta" "$supplement_bed")"; then
        validation="${out_dir}/download_error.txt"
        printf "%s\n" "$download_reason" > "$validation"
        printf "%s\t%s\tdownload_failed\t\t\t%s\t\t\t\n" "$code" "$species" "$validation" >> "$PANEL_TSV"
        print_species_line "$code" "$species" "download_failed" "" "" "" "$download_reason"
        failed_count=$((failed_count + 1))
        continue
    fi

    genome="${out_dir}/genome.fa"
    raw_bed="${out_dir}/${code}-precursors.bed"
    bed="${out_dir}/${code}-precursors.normalized.bed"
    alias="${out_dir}/chromAlias.txt"
    normalize_report="${out_dir}/bed_chrom_normalization.txt"
    validation="${out_dir}/bed_genome_validation.txt"

    if [ ! -s "$genome" ] || [ ! -s "$raw_bed" ]; then
        printf "missing required downloaded files\n" > "$validation"
        printf "%s\t%s\tdownload_failed\t%s\t%s\t%s\t\t\t\n" "$code" "$species" "$genome" "$raw_bed" "$validation" >> "$PANEL_TSV"
        print_species_line "$code" "$species" "download_failed" "" "" "" "missing required downloaded files"
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
        printf "%s\t%s\tnormalization_failed\t%s\t%s\t%s\t%s\t%s\t%s\n" "$code" "$species" "$genome" "$raw_bed" "$normalize_report" "$bed_rows" "$matched_rows" "$dropped_rows" >> "$PANEL_TSV"
        print_species_line "$code" "$species" "normalization_failed" "$bed_rows" "$matched_rows" "$dropped_rows"
        failed_count=$((failed_count + 1))
        continue
    fi

    bed_rows="$(report_value "$normalize_report" "bed_rows")"
    matched_rows="$(report_value "$normalize_report" "matched_rows")"
    dropped_rows="$(report_value "$normalize_report" "dropped_rows")"
    if python "$VALIDATOR" --bed "$bed" --genome "$genome" > "$validation" 2>&1; then
        printf "%s\t%s\tauto\t%s\t%s\t%s\t%s\t%s\t%s\n" "$code" "$species" "$genome" "$bed" "$validation" "$bed_rows" "$matched_rows" "$dropped_rows" >> "$PANEL_TSV"
        print_species_line "$code" "$species" "auto" "$bed_rows" "$matched_rows" "$dropped_rows"
        success_count=$((success_count + 1))
    else
        printf "%s\t%s\tvalidation_failed\t%s\t%s\t%s\t%s\t%s\t%s\n" "$code" "$species" "$genome" "$bed" "$validation" "$bed_rows" "$matched_rows" "$dropped_rows" >> "$PANEL_TSV"
        print_species_line "$code" "$species" "validation_failed" "$bed_rows" "$matched_rows" "$dropped_rows"
        failed_count=$((failed_count + 1))
    fi
done < <(
    python - "$PANEL_SOURCE" <<'PY'
import csv
import sys

fields = [
    "code",
    "common_name",
    "scientific_name",
    "taxonomy_group",
    "pre_mirna_count",
    "genome_source",
    "assembly_name",
    "assembly_accession",
    "genome_fasta_url",
    "alias_url",
    "mirgenedb_bed_url",
    "supplement_fasta",
    "supplement_bed",
    "panel_status",
    "notes",
]

with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        print("\034".join(row.get(field, "") for field in fields))
PY
)

echo "panel manifest: $PANEL_TSV"
echo "auto species: $success_count"
echo "failed species: $failed_count"
