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

### Conda

```bash
./setup.sh --tool {tool}
```

### Docker

```bash
./setup.sh --tool {tool} --docker
```

Download test data:

```bash
cd {tool}
./download_test_data.sh
```

## Run Inference

### Conda Mode

```bash
python inference.py --tool {tool} --output-name my_run
```

### Docker Mode

```bash
python inference.py --tool {tool} --docker --output-name my_run
```

## Configuration

Edit `configs/{tool}_config.yaml` to set parameters.

## Results

Results saved to `../results/{tool}/{output-name}/`.
