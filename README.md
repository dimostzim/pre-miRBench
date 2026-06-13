# pre-miRBench

Pipeline for retraining and evaluating pre-miRNA prediction tools.

## Structure

- `pipeline/download_data.sh` - download the final species panel and write `panel.tsv`
- `pipeline/build_dataset.py` - build canonical train/validation/test splits and per-tool inputs
- `pipeline/train.py` - train one tool image on the prepared dataset
- `pipeline/evaluate.py` - evaluate trained tools on held-out splits
- `tools/<tool>/` - Dockerfile plus tool-specific `train.py` and `inference.py` adapters
- `tests/fixtures/` - small committed fixtures used by tests

Generated data and outputs are ignored:

- `data/raw/`
- `data/datasets/`
- `data/work/`
- `results/training/`
- `results/evaluation/`
- `tools/*/*_src/`

## Workflow

Download the species panel:

```bash
bash pipeline/download_data.sh
```

Build the retraining dataset:

```bash
python pipeline/build_dataset.py
```

Build Docker images:

```bash
bash tools/setup_images.sh
```

Train each tool:

```bash
for tool in deepmir deepmirgene dnnpremir mirdnn mire2e mustard; do
  python pipeline/train.py --tool "$tool" --run-name diverse20_gpu_1to5
done
```

Evaluate trained tools:

```bash
python pipeline/evaluate.py --run-name diverse20_gpu_1to5
```
