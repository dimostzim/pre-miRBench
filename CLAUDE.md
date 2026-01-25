# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a unified wrapper repository for benchmarking multiple pre-miRNA prediction tools. It provides consistent setup and inference interfaces for six different prediction tools: mustard, mire2e, mirdnn, dnnpremir, deepmir, and deepmirgene.

## Common Commands

### Setup

Install a tool's conda environment:
```bash
./setup.sh --tool {tool}
```

Build Docker image (use on ARM/macOS for compatibility):
```bash
./setup.sh --tool {tool} --docker
```

Download test data for a specific tool:
```bash
cd tools/{tool}
./download_test_data.sh
```

### Running Inference

Conda mode (uses conda environment):
```bash
python inference.py --tool {tool} --output-name my_run
```

Docker mode (platform-agnostic):
```bash
python inference.py --tool {tool} --docker --output-name my_run
```

Results are saved to `results/{tool}/{output-name}/`.

### Configuration

Tool-specific parameters are set in `configs/{tool}_config.yaml`. Edit these files to change parameters like input paths, model selection, batch size, device (cpu/gpu), etc.

## Architecture

### Main Components

1. **Root-level unified interface** (`inference.py`, `setup.sh`)
   - `inference.py`: Wrapper that reads tool-specific config files, constructs appropriate command-line arguments, and invokes tool-specific inference scripts
   - Supports both conda and Docker execution modes
   - Handles path translation for Docker containers (`/work/` prefix)

2. **Tool-specific implementations** (`tools/{tool}/`)
   - Each tool has its own directory with:
     - `inference.py`: Tool-specific inference script
     - `setup.sh`: Tool-specific environment setup
     - `environment.yml`: Conda dependencies
     - `Dockerfile`: Docker build configuration
     - `data/`: Test data and pre-trained models
   - Tools are independent wrappers around various ML architectures (CNN, RNN, Transformer)

3. **Folding pipeline** (`folding/`)
   - RNA secondary structure folding and analysis
   - `run_folding.py`: Main pipeline that creates sliding windows, runs RNAfold, and analyzes results
   - Scripts:
     - `make_windows.py`: Creates sliding windows from input sequences
     - `run_fold.py`: Runs RNAfold on windows
     - `analyze_fold.py`: Analyzes folding energy distributions
     - `sample_negatives.py`: Generates negative training examples
     - `train_rf.py`: Trains Random Forest classifier using sequence, structure, and MFE features
     - `find_mirna_windows.py`: Identifies windows containing miRNA sequences

4. **Orthologues pipeline** (`premirna_orthologues/`)
   - `get_orthologues.py`: Fetches miRNA orthologue data (web scraping with rate limiting)
   - `scrape_orthologues.py`: Additional scraping utilities

### Execution Flow

1. User configures parameters in `configs/{tool}_config.yaml`
2. User runs `inference.py --tool {tool} --output-name {name}`
3. Root `inference.py` loads config, determines execution mode (conda vs docker)
4. Constructs tool-specific command with appropriate arguments
5. Invokes `tools/{tool}/inference.py` with environment activated
6. Tool-specific script loads model and runs predictions
7. Results saved to `results/{tool}/{output-name}/`

### Key Design Patterns

- **Unified interface**: All tools accept similar high-level arguments but have different underlying implementations
- **Config-driven**: Each tool has a YAML config with required and optional parameters
- **Dual execution modes**: Conda for performance, Docker for portability (especially ARM compatibility)
- **Path abstraction**: Root inference script handles path translation between host and container environments

## Tool-Specific Notes

### MuStARD
- Multi-scale CNN with structure and conservation
- Requires: target intervals (BED), genome (FASTA), conservation data (wigFix)
- Perl-based with Python ML components
- Multiple pre-trained models available (mirS, mirF, mirSFC, mirSFC-U, etc.)

### MiRe2e
- End-to-end Transformer model
- Pretrained models: hsa (human), mmu (mouse), or custom
- Outputs bidirectional scores (5'→3' and 3'→5')

### MirDNN, DNNPremir, DeepMir, DeepMiRGene
- Various CNN/RNN architectures with structure inputs
- Each has different sequence length requirements
- Some use image-based representations of RNA structure

## Development Notes

- All tools run in CPU mode when using Docker (no GPU support in Docker on macOS)
- Docker uses `--platform linux/amd64` for ARM compatibility via emulation
- PyYAML is required at the root level for the unified interface
- Each tool maintains its own dependencies via conda or Docker
