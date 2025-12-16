#!/bin/bash
set -e
cd "$(dirname "$0")"

mkdir -p data

# Copy test data from cloned repository
if [ -d "deepmir_src/examples" ]; then
    echo "Copying test data from deepmir_src..."
    cp deepmir_src/examples/sequences.fasta data/test.fa
    echo "Test data copied to data/test.fa"
else
    echo "Error: deepmir_src/examples not found. Please run setup.sh first."
    exit 1
fi
