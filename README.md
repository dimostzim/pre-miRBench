# Pre-miRNA Prediction Pipeline

Pipeline for pre-miRNA prediction: benchmark dataset creation, tool inference, and evaluation.

## Structure

- `benchmark/` - Benchmarking pipeline
  - `balanced_benchmark/` - Collapsed balanced tool benchmark implementation and configs
  - `download/` - Data download scripts
  - `fold/` - RNA folding pipeline
  - `make_negative_set/` - Balanced negative sampling
- `tools/` - Pre-miRNA prediction tool wrappers
- `configs/` - Tool configuration files

## Documentation

- `benchmark/README.md` - Benchmark pipeline, including the full collapsed balanced benchmark
- `benchmark/balanced_benchmark/COLLAPSE.md` - How collapsed positive windows are selected
- `benchmark/download/README.md` - Data download
- `tools/README.md` - Tool benchmarking
