#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p data

wget -O data/test.fold "https://raw.githubusercontent.com/cyones/mirDNN/master/sequences/test.fold"
# Convert test.fold to test.fa by extracting headers and sequences
awk 'NR%3==1 {print} NR%3==2 {print}' data/test.fold > data/test.fa
