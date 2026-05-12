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

## Retrain Models

Retraining uses Docker and writes artifacts under `results/training/{tool}/{run-name}/`:

```bash
python tools/train.py --tool {tool} --run-name my_model --config configs/train/{tool}_train.yaml
```

The training configs are examples for user-supplied inputs. FASTA-native tools
use positive/negative FASTA files; dnnPreMiR folds FASTA inputs and generates
its `seq_struc` CSV representation unless precomputed CSVs are supplied; miRe2e
can generate its structure/MFE training representation and retrain all three
stages. MuStARD uses its native interval/genome/conservation inputs.
After a successful run, `inference_config.yaml` is written in the training
result directory and can be passed to `tools/inference.py`.

## Configuration

Edit `configs/{tool}_config.yaml` to set default parameters, or pass an explicit config file with `--config`.

The root configs now point at the prepared balanced benchmark inputs under
`benchmark/prepared_inputs/balanced_benchmark/{tool}/`, so run
`python benchmark/balanced_benchmark/prepare_inputs.py` first if you want to use
those defaults. Otherwise pass an explicit config file with `--config`.

## Results

Results are written to `results/{tool}/{output-name}/`.
