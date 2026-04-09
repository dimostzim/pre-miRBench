# How to Use pre-miRBench

Unified benchmarking pipeline for six pre-miRNA prediction tools:
**mustard**, **mire2e**, **mirdnn**, **dnnpremir**, **deepmir**, **deepmirgene**

---

## Prerequisites

- **Docker Desktop** (required; all tools run in containers for reproducibility on ARM/macOS)
- **Python 3.9+** with `pyyaml`, `biopython`, `scikit-learn`, `matplotlib` at the repo root level
- The repo root must be your working directory for all commands below

---

## 1. Build Tool Images

Build each tool's Docker image once before running:

```bash
cd tools/mire2e    && docker build -t mire2e:latest .
cd tools/mirdnn    && docker build -t mirdnn:latest .
cd tools/dnnpremir && docker build -t dnnpremir:latest .
cd tools/deepmir   && docker build -t deepmir:latest .
cd tools/deepmirgene && docker build -t deepmirgene:latest .
cd tools/mustard   && docker build -t mustard:latest .
```

> On ARM/macOS, images use `--platform linux/amd64` automatically via the orchestrator. This is handled at runtime — no extra flags needed when building.

---

## 2. Prepare a Dataset

### Option A — Use the included C. elegans balanced dataset

A ready-to-use balanced dataset (138 positives, 138 negatives) is at:

```
benchmark/output/sample_negatives_output/balanced_collapsed.csv
```

Run `prepare_inputs.py` to generate per-tool FASTA and BED files:

```bash
python3 benchmark/balanced_benchmark/prepare_inputs.py \
  --input   benchmark/output/sample_negatives_output/balanced_collapsed.csv \
  --output-dir benchmark/prepared_inputs/cel_balanced \
  --prefix  cel_balanced \
  --truth-bed benchmark/download/data/cel-precursors-no-v2.bed
```

This produces `benchmark/prepared_inputs/cel_balanced/{tool}/cel_balanced.{fa,bed,metadata.csv}` for all six tools.

### Option B — Bring your own dataset

Your CSV must have these columns:

| Column | Description |
|--------|-------------|
| `window_id` | Unique ID, e.g. `chrIV\|1021251-1021450\|-` |
| `chrom` | Chromosome name |
| `start` / `end` | 1-based genomic coordinates |
| `strand` | `+`, `-`, or `.` |
| `sequence` | RNA sequence (U or T, will be normalised to U) |
| `label` | `positive` or `negative` |
| `target_mirna` | Name matching the BED file (for positives; blank for negatives) |
| *(optional)* `structure`, `mfe`, `contained_mirnas`, `num_mirnas` | |

Then run `prepare_inputs.py` as above, substituting your file and a matching `--truth-bed`.

---

## 3. Write Tool Configs

Create one YAML config per tool. Example configs for the C. elegans run are in `benchmark/balanced_benchmark/configs/cel_balanced/`. The minimum required fields per tool:

**mire2e**
```yaml
input: benchmark/prepared_inputs/cel_balanced/mire2e/cel_balanced.fa
device: cpu
pretrained: hsa        # hsa | mmu | animals
length: 100
step: 20
batch_size: 4096
```

**mirdnn**
```yaml
input: benchmark/prepared_inputs/cel_balanced/mirdnn/cel_balanced.fa
model: animal          # animal | plants
device: cpu
seq_length: 160
batch_size: 1024
```

**dnnpremir**
```yaml
input: benchmark/prepared_inputs/cel_balanced/dnnpremir/cel_balanced.fa
```

**deepmir**
```yaml
input: benchmark/prepared_inputs/cel_balanced/deepmir/cel_balanced.fa
model: fine_tuned_cnn
```

**deepmirgene**
```yaml
input: benchmark/prepared_inputs/cel_balanced/deepmirgene/cel_balanced.fa
model: null
```

**mustard** (sequence + RNAfold, no conservation)
```yaml
targetIntervals: benchmark/prepared_inputs/cel_balanced/mustard/cel_balanced.bed
genome: benchmark/download/data/ce11.fa
consDir: benchmark/download/data
chromList: chrI,chrII,chrIII,chrIV,chrV,chrX
model: MuStARD-mirSF     # use MuStARD-mirSFC-U if PhyloP wigFix files are available
classNum: 2
modelType: CNN
winSize: 100
step: 5
staticPredFlag: 1
inputMode: sequence,RNAfold
threads: 10
modelDirName: results
intermDir: results/mustard_intermediate
```

For mustard with conservation, set `model: MuStARD-mirSFC-U`, `inputMode: sequence,RNAfold,conservation`, and place `{chrom}.wigFix.gz` files in `consDir`.

---

## 4. Run Inference

Run from the **repo root**. Each tool writes `results/{tool}/{output-name}/predictions.csv`.

```bash
python3 tools/inference.py --tool mire2e     --output-name cel_balanced --config benchmark/balanced_benchmark/configs/cel_balanced/mire2e.yaml
python3 tools/inference.py --tool mirdnn     --output-name cel_balanced --config benchmark/balanced_benchmark/configs/cel_balanced/mirdnn.yaml
python3 tools/inference.py --tool dnnpremir  --output-name cel_balanced --config benchmark/balanced_benchmark/configs/cel_balanced/dnnpremir.yaml
python3 tools/inference.py --tool deepmir    --output-name cel_balanced --config benchmark/balanced_benchmark/configs/cel_balanced/deepmir.yaml
python3 tools/inference.py --tool deepmirgene --output-name cel_balanced --config benchmark/balanced_benchmark/configs/cel_balanced/deepmirgene.yaml
python3 tools/inference.py --tool mustard    --output-name cel_balanced --config benchmark/balanced_benchmark/configs/cel_balanced/mustard.yaml
```

All tools run in Docker automatically; no conda activation needed.

### Unified output format

Every tool produces the same schema regardless of its internal output format:

```
results/{tool}/{output-name}/predictions.csv
window_id,probability_score
cel_balanced__000001,0.9956
cel_balanced__000002,0.0001
...
```

`window_id` here is the internal `record_id` (`{prefix}__{i:06d}`). The metadata file maps it back to the original genomic window.

---

## 5. Evaluate

```bash
python3 benchmark/balanced_benchmark/evaluate_outputs.py \
  --prepared-dir benchmark/prepared_inputs/cel_balanced \
  --results-dir  results \
  --output-dir   benchmark/evaluated/cel_balanced \
  --prefix       cel_balanced \
  --tools        all \
  --threshold    0.5
```

This produces:

| File | Contents |
|------|----------|
| `benchmark/evaluated/cel_balanced/{tool}.csv` | Per-record predictions + ground truth |
| `benchmark/evaluated/cel_balanced/metrics.csv` | TP, FP, TN, FN, precision, recall, F1, MCC, ROC AUC, PR AUC per tool |
| `benchmark/evaluated/cel_balanced/curves.json` | ROC and PR curve data for plotting |

To evaluate only specific tools:
```bash
--tools mire2e,mirdnn,deepmirgene
```

---

## 6. Plot Results

```bash
python3 benchmark/balanced_benchmark/plot_metrics.py \
  --metrics-csv benchmark/evaluated/cel_balanced/metrics.csv \
  --curves-json benchmark/evaluated/cel_balanced/curves.json \
  --out-dir     benchmark/evaluated/cel_balanced \
  --title       "C. elegans Pre-miRNA Benchmark (Balanced)"
```

Produces three figures:

| File | Description |
|------|-------------|
| `metrics_4panel.png` | Precision, recall, F1, MCC bar charts per tool |
| `auc_curves.png` | ROC and PR curves for all tools |
| `comparison_table.png` | Sortable summary table of all metrics |

---

## Example: C. elegans end-to-end

```bash
# 1. Prepare inputs
python3 benchmark/balanced_benchmark/prepare_inputs.py \
  --input benchmark/output/sample_negatives_output/balanced_collapsed.csv \
  --output-dir benchmark/prepared_inputs/cel_balanced \
  --prefix cel_balanced \
  --truth-bed benchmark/download/data/cel-precursors-no-v2.bed

# 2. Run all 6 tools
for TOOL in mire2e mirdnn dnnpremir deepmir deepmirgene mustard; do
  python3 tools/inference.py --tool $TOOL --output-name cel_balanced \
    --config benchmark/balanced_benchmark/configs/cel_balanced/${TOOL}.yaml
done

# 3. Evaluate
python3 benchmark/balanced_benchmark/evaluate_outputs.py \
  --prepared-dir benchmark/prepared_inputs/cel_balanced \
  --results-dir results \
  --output-dir benchmark/evaluated/cel_balanced \
  --prefix cel_balanced --tools all

# 4. Plot
python3 benchmark/balanced_benchmark/plot_metrics.py \
  --metrics-csv benchmark/evaluated/cel_balanced/metrics.csv \
  --curves-json benchmark/evaluated/cel_balanced/curves.json \
  --out-dir benchmark/evaluated/cel_balanced \
  --title "C. elegans Pre-miRNA Benchmark (Balanced)"
```

---

## File Structure

```
pre-miRBench/
├── tools/
│   ├── inference.py                  # Orchestrator — runs any tool via Docker
│   ├── {tool}/
│   │   ├── Dockerfile
│   │   ├── inference.py              # Tool-specific inference script (run inside Docker)
│   │   ├── patch_upstream.py         # Build-time source patches (deepmir, dnnpremir, deepmirgene)
│   │   └── runtime_predictor.py      # Runtime override (deepmir only)
├── benchmark/
│   ├── balanced_benchmark/
│   │   ├── prepare_inputs.py         # Generate per-tool FASTA/BED from dataset CSV
│   │   ├── evaluate_outputs.py       # Compute metrics from predictions + metadata
│   │   ├── plot_metrics.py           # Generate plots from metrics.csv + curves.json
│   │   ├── tool_adapters.py          # Shared utilities: sequence cropping, ID mapping, normalisation
│   │   ├── metrics.py                # ROC AUC, PR AUC via sklearn
│   │   └── configs/
│   │       └── cel_balanced/         # Per-tool YAML configs for the C. elegans run
│   ├── output/
│   │   ├── fold_output/              # RNAfold output for C. elegans genome windows
│   │   └── sample_negatives_output/  # Balanced dataset (positives + MFE-matched negatives)
│   └── download/data/                # ce11.fa, cel-precursors-no-v2.bed, etc.
├── results/
│   └── {tool}/{output-name}/
│       └── predictions.csv           # Unified output: window_id, probability_score
└── benchmark/evaluated/
    └── {output-name}/
        ├── metrics.csv
        ├── curves.json
        └── *.png
```

---

## Notes

- **All commands must be run from the repo root** (`pre-miRBench/`), not from within subdirectories.
- Re-running a tool overwrites `predictions.csv`. Previous results in other output names are unaffected.
- MuStARD without PhyloP conservation files (`MuStARD-mirSF`) will score ~0.5 AUC on cross-species data; this is expected. For a fair comparison include conservation files and use `MuStARD-mirSFC-U`.
- Docker images are built once and reused. If you modify `tools/{tool}/inference.py` after building, the change takes effect immediately (the orchestrator volume-mounts the script at runtime for mire2e, mirdnn; deepmir uses `runtime_predictor.py`). For dnnpremir, deepmirgene, and mustard the inference scripts run from the image — rebuild if you change them.
