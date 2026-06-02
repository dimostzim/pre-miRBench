# Pre-miRNA Prediction Pipeline

Pipeline for pre-miRNA prediction: training dataset creation, tool inference,
and tool retraining.

## Structure

- `benchmark/` - Data download and training dataset pipeline
  - `download/` - Data download scripts
  - `train_data/` - Positive extraction, negative mining, splits, and tool input preparation
- `tools/` - Pre-miRNA prediction tool wrappers
- `configs/` - Tool configuration files

## Documentation

- `benchmark/README.md` - Training data pipeline
- `benchmark/download/README.md` - Data download
- `tools/README.md` - Tool benchmarking
