# Pre-miRNA Prediction Tools

Unified wrapper for benchmarking pre-miRNA prediction tools.

## Available Tools

- **mustard** - Multi-scale CNN with structure and conservation
- **mire2e** - End-to-end Transformer
- **mirdnn** - Hybrid CNN/RNN with structure
- **dnnpremir** - CNN with structure
- **deepmir** - Image-based CNN with structure
- **deepmirgene** - RNN with attention and structure

## Setup

Build a tool image from the repository root:

```bash
cd tools
./setup.sh --tool {tool}
cd ..
```

Download a tool's bundled smoke-test data if needed:

```bash
cd tools/{tool}
./download_test_data.sh
cd ../..
```

## Run Inference

Run from the repository root:

```bash
python tools/inference.py --tool {tool} --output-name my_run
```

## Configuration

Edit `configs/{tool}_config.yaml` to set default parameters, or pass an explicit config file with `--config`.

## Results

Results are written to `results/{tool}/{output-name}/`.
