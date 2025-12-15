#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p data

# convert test.fold to test.fa by extracting headers and sequences
awk 'NR%3==1 {print} NR%3==2 {print}' mirdnn_src/sequences/test.fold > data/test.fa

echo "Test data prepared in data/test.fa"
