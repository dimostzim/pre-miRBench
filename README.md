# Pre-miRNA Prediction Pipeline

Pipeline for pre-miRNA prediction: benchmark dataset creation, tool inference, and evaluation.

## Structure

- `benchmark/` - Benchmarking pipeline
  - `download/` - Data download scripts
  - `fold/` - RNA folding pipeline
  - `make_negative_set/` - Balanced negative sampling
  - `models/` - Model training scripts
  - `prepare_1_1_inputs.py` / `evaluate_1_1_outputs.py` - collapsed 1:1 tool benchmark
- `tools/` - Pre-miRNA prediction tool wrappers
- `configs/` - Tool configuration files

## Documentation

- `benchmark/README.md` - Benchmark pipeline, including the full collapsed `1_1` tool benchmark
- `benchmark/download/README.md` - Data download
- `tools/README.md` - Tool benchmarking
