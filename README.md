# Pre-miRNA Prediction Pipeline

Pipeline for pre-miRNA prediction: data preparation, RNA folding analysis, balanced dataset creation, and model training.

## Structure

- `benchmark/` - Benchmarking pipeline
  - `download/` - Data download scripts
  - `fold/` - RNA folding pipeline
  - `sample_negatives/` - Balanced negative sampling
  - `models/` - Model training scripts
- `tools/` - Pre-miRNA prediction tool wrappers
- `configs/` - Tool configuration files

## Documentation

- `benchmark/README.md` - Benchmark pipeline (folding, negatives, models)
- `benchmark/download/README.md` - Data download
- `tools/README.md` - Tool benchmarking
