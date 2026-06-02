#!/bin/bash
set -euo pipefail

SPECIES_CODES="${1:-hsa,mmu,rno,dme,dre,cel}"

printf "%-8s %-8s %-12s %s\n" "code" "http" "pre_no_v2" "url"
IFS=',' read -r -a codes <<< "$SPECIES_CODES"
for code in "${codes[@]}"; do
    code="${code// /}"
    [ -z "$code" ] && continue
    url="https://mirgenedb.org/static/data/${code}/${code}-all.bed"
    tmp_file="$(mktemp)"
    status="$(curl -L -s -o "$tmp_file" -w "%{http_code}" "$url" || true)"
    if [ "$status" = "200" ]; then
        count="$(awk '$4 ~ /_pre$/' "$tmp_file" | grep -v -- "-v2_" | wc -l | tr -d ' ')"
    else
        count="0"
    fi
    rm -f "$tmp_file"
    printf "%-8s %-8s %-12s %s\n" "$code" "$status" "$count" "$url"
done
