#!/bin/bash
set -e
cd "$(dirname "$0")"

mkdir -p data

# Copy test data from cloned repository
if [ -d "dnnpremir_src/testData" ]; then
    echo "Copying test data from dnnpremir_src..."
    cp dnnpremir_src/testData/hsa-pre-release22.fa data/test.fa
    echo "Test data copied to data/test.fa"
else
    echo "Error: dnnpremir_src/testData not found. Please run setup.sh first."
    exit 1
fi
