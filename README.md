# Pre-miRNA Prediction Pipeline

Pipeline for pre-miRNA prediction: benchmark dataset creation, tool inference, and evaluation.

## Structure

- `benchmark/` - Benchmarking pipeline
  - `1_1/` - Collapsed 1:1 tool benchmark implementation and configs
  - `download/` - Data download scripts
  - `fold/` - RNA folding pipeline
  - `make_negative_set/` - Balanced negative sampling
  - `models/` - Model training scripts
- `tools/` - Pre-miRNA prediction tool wrappers
- `configs/` - Tool configuration files

## Documentation

- `benchmark/README.md` - Benchmark pipeline, including the full collapsed `1_1` tool benchmark
- `benchmark/1_1/COLLAPSE.md` - How collapsed positive windows are selected
- `benchmark/download/README.md` - Data download
- `tools/README.md` - Tool benchmarking
