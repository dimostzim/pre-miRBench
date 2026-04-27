# How to Use pre-miRBench

Unified benchmarking pipeline for six pre-miRNA prediction tools:
**mustard**, **mire2e**, **mirdnn**, **dnnpremir**, **deepmir**, **deepmirgene**

---

## Prerequisites

Before you start, make sure you have the following installed:

1. **Docker Desktop** — all six tools run inside Docker containers. Download from [docker.com](https://www.docker.com/products/docker-desktop/) and make sure it is running before proceeding.
2. **Python 3.9 or later** — used for the prepare, evaluate, and plot scripts.
3. **Python packages** — install once at the repo root:
   ```bash
   pip install pyyaml biopython scikit-learn matplotlib
   ```
4. **Clone the repo and enter it:**
   ```bash
   git clone https://github.com/dimostzim/pre-miRBench.git
   cd pre-miRBench
   ```

> **Important:** every command in this guide must be run from the **repo root** (`pre-miRBench/`), not from any subdirectory.

---

## Step 1 — Build the Docker Images

Each tool has its own Docker image. Build them once; they are reused on every subsequent run.

```bash
(cd tools/mire2e     && docker build -t mire2e:latest .)
(cd tools/mirdnn     && docker build -t mirdnn:latest .)
(cd tools/dnnpremir  && docker build -t dnnpremir:latest .)
(cd tools/deepmir    && docker build -t deepmir:latest .)
(cd tools/deepmirgene && docker build -t deepmirgene:latest .)
(cd tools/mustard    && docker build -t mustard:latest .)
```

Each build takes several minutes the first time (conda environments are created inside). Subsequent runs use the Docker layer cache and are instant.

> On Apple Silicon (M1/M2/M3) and other ARM machines, the orchestrator automatically adds `--platform linux/amd64` at runtime — no extra flags are needed when building.

---

## Step 2 — Prepare the Dataset Inputs

This step converts the dataset CSV into per-tool FASTA and BED files, resizing sequences to each tool's required window length.

### Using the included C. elegans balanced dataset

A ready-to-use balanced dataset (138 positives, 138 negatives) is already in the repo:

```
benchmark/output/sample_negatives_output/balanced_collapsed.csv
```

Run the preparation script from the repo root:

```bash
python3 benchmark/balanced_benchmark/prepare_inputs.py \
  --input      benchmark/output/sample_negatives_output/balanced_collapsed.csv \
  --output-dir benchmark/prepared_inputs/cel_balanced \
  --prefix     cel_balanced \
  --truth-bed  benchmark/download/data/cel-precursors-no-v2.bed
```

This creates one subdirectory per tool under `benchmark/prepared_inputs/cel_balanced/`, each containing:
- `cel_balanced.fa` — FASTA sequences resized to the tool's required window length
- `cel_balanced.bed` — BED intervals (used by mustard)
- `cel_balanced.metadata.csv` — record ID ↔ original window ID ↔ label mapping

### Bringing your own dataset

Your input CSV must have these columns:

| Column | Description |
|--------|-------------|
| `window_id` | Unique identifier, e.g. `chrIV\|1021251-1021450\|-` |
| `chrom` | Chromosome name |
| `start` / `end` | 1-based genomic coordinates |
| `strand` | `+`, `-`, or `.` |
| `sequence` | RNA sequence (T or U; normalised to U automatically) |
| `label` | `positive` or `negative` |
| `target_mirna` | Name matching the `--truth-bed` file (positives only; leave blank for negatives) |
| *(optional)* | `structure`, `mfe`, `contained_mirnas`, `num_mirnas` |

Run `prepare_inputs.py` as above, substituting your file paths.

---

## Step 3 — Create Tool Config Files

Each tool is configured with a small YAML file. Config files for the C. elegans run are already provided in `benchmark/balanced_benchmark/configs/cel_balanced/`. You can copy and edit them for your own dataset.

The minimum fields required per tool are shown below. Paths must be relative to the repo root.

**mire2e** (`configs/cel_balanced/mire2e.yaml`)
```yaml
input: benchmark/prepared_inputs/cel_balanced/mire2e/cel_balanced.fa
device: cpu
pretrained: hsa        # hsa | mmu | animals
length: 100
step: 20
batch_size: 4096
```

**mirdnn** (`configs/cel_balanced/mirdnn.yaml`)
```yaml
input: benchmark/prepared_inputs/cel_balanced/mirdnn/cel_balanced.fa
model: animal          # animal | plants
device: cpu
seq_length: 160
batch_size: 1024
```

**dnnpremir** (`configs/cel_balanced/dnnpremir.yaml`)
```yaml
input: benchmark/prepared_inputs/cel_balanced/dnnpremir/cel_balanced.fa
```

**deepmir** (`configs/cel_balanced/deepmir.yaml`)
```yaml
input: benchmark/prepared_inputs/cel_balanced/deepmir/cel_balanced.fa
model: fine_tuned_cnn
```

**deepmirgene** (`configs/cel_balanced/deepmirgene.yaml`)
```yaml
input: benchmark/prepared_inputs/cel_balanced/deepmirgene/cel_balanced.fa
model: null
```

**mustard** (`configs/cel_balanced/mustard.yaml`)
```yaml
targetIntervals: benchmark/prepared_inputs/cel_balanced/mustard/cel_balanced.bed
genome:   benchmark/download/data/ce11.fa
consDir:  benchmark/download/data
chromList: chrI,chrII,chrIII,chrIV,chrV,chrX
model: MuStARD-mirSF        # use MuStARD-mirSFC-U when PhyloP wigFix files are available
classNum: 2
modelType: CNN
winSize: 100
step: 5
staticPredFlag: 1
inputMode: sequence,RNAfold  # add ,conservation when PhyloP files are present
threads: 10
modelDirName: results
intermDir: results/mustard_intermediate
```

---

## Step 4 — Run Inference

Run each tool from the repo root. The `--output-name` argument sets the subfolder name under `results/{tool}/` where outputs are saved.

```bash
python3 tools/inference.py --tool mire2e      --output-name cel_balanced \
  --config benchmark/balanced_benchmark/configs/cel_balanced/mire2e.yaml

python3 tools/inference.py --tool mirdnn      --output-name cel_balanced \
  --config benchmark/balanced_benchmark/configs/cel_balanced/mirdnn.yaml

python3 tools/inference.py --tool dnnpremir   --output-name cel_balanced \
  --config benchmark/balanced_benchmark/configs/cel_balanced/dnnpremir.yaml

python3 tools/inference.py --tool deepmir     --output-name cel_balanced \
  --config benchmark/balanced_benchmark/configs/cel_balanced/deepmir.yaml

python3 tools/inference.py --tool deepmirgene --output-name cel_balanced \
  --config benchmark/balanced_benchmark/configs/cel_balanced/deepmirgene.yaml

python3 tools/inference.py --tool mustard     --output-name cel_balanced \
  --config benchmark/balanced_benchmark/configs/cel_balanced/mustard.yaml
```

Each command launches a Docker container, runs the model, and saves results to `results/{tool}/cel_balanced/`. No conda activation is needed.

### Output format options — `--norm-output`

The `--norm-output` flag controls whether outputs are written in the **unified benchmark format** or the **original native format** of each tool.

| Flag | Behaviour |
|------|-----------|
| `--norm-output y` | *(default)* Every tool writes `predictions.csv` with columns `window_id, probability_score`. Required for the evaluate and plot steps. |
| `--norm-output n` | Each tool writes its original output format (see table below). Useful if you need the native outputs for downstream analysis. |

**Native formats produced by `--norm-output n`:**

| Tool | File | Format |
|------|------|--------|
| mire2e | `predictions.json` | JSON list of `{window, score_5_3, score_3_5}` per sub-window |
| mirdnn | `predictions.csv` | CSV with **no header row** — `id, score` |
| dnnpremir | `predictions.txt` | Block text: `>id`, `SEQUENCE  True/False`, `===` separator |
| deepmir | `results.csv` | CSV with columns `hairpin, sequence, fold, label` |
| deepmirgene | `predictions.txt` | One integer per line (`1` = pre-miRNA, `0` = other) |
| mustard | `targets.{chrom}.predictions.txt.gz` | Gzip TSV, two float columns per interval (class 0, class 1), one file per chromosome |

Example — run mirdnn in native format:
```bash
python3 tools/inference.py --tool mirdnn --output-name cel_raw \
  --config benchmark/balanced_benchmark/configs/cel_balanced/mirdnn.yaml \
  --norm-output n
```

> The evaluate and plot steps (Steps 5–6) require `--norm-output y` (the default).

---

## Step 5 — Evaluate

Once all tools have run with `--norm-output y`, compute metrics:

```bash
python3 benchmark/balanced_benchmark/evaluate_outputs.py \
  --prepared-dir benchmark/prepared_inputs/cel_balanced \
  --results-dir  results \
  --output-dir   benchmark/evaluated/cel_balanced \
  --prefix       cel_balanced \
  --tools        all \
  --threshold    0.5
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--prepared-dir` | Directory created by `prepare_inputs.py` |
| `--results-dir` | Root of the results tree (default: `results/`) |
| `--output-dir` | Where to write evaluation outputs |
| `--prefix` | Must match the prefix used in Step 2 |
| `--tools` | `all` or a comma-separated subset, e.g. `mire2e,mirdnn` |
| `--threshold` | Score threshold for binary predictions (default `0.5`) |

**Output files:**

| File | Contents |
|------|----------|
| `{tool}.csv` | Per-record scores, predicted class, and ground-truth label |
| `metrics.csv` | TP, FP, TN, FN, precision, recall, F1, MCC, ROC AUC, PR AUC for every tool |
| `curves.json` | Full ROC and PR curve data for plotting |

---

## Step 6 — Plot Results

```bash
python3 benchmark/balanced_benchmark/plot_metrics.py \
  --metrics-csv benchmark/evaluated/cel_balanced/metrics.csv \
  --curves-json benchmark/evaluated/cel_balanced/curves.json \
  --out-dir     benchmark/evaluated/cel_balanced \
  --title       "C. elegans Pre-miRNA Benchmark (Balanced)"
```

Three figures are written to `--out-dir`:

| File | Description |
|------|-------------|
| `metrics_4panel.png` | Bar charts for precision, recall, F1, and MCC per tool |
| `auc_curves.png` | ROC and precision-recall curves for all tools |
| `comparison_table.png` | Summary table of all metrics, sortable by column |

---

## Full End-to-End Example (C. elegans)

Copy and run these commands from the repo root to reproduce the complete C. elegans benchmark:

```bash
# Step 1: build images (skip if already built)
(cd tools/mire2e     && docker build -t mire2e:latest .)
(cd tools/mirdnn     && docker build -t mirdnn:latest .)
(cd tools/dnnpremir  && docker build -t dnnpremir:latest .)
(cd tools/deepmir    && docker build -t deepmir:latest .)
(cd tools/deepmirgene && docker build -t deepmirgene:latest .)
(cd tools/mustard    && docker build -t mustard:latest .)

# Step 2: prepare inputs
python3 benchmark/balanced_benchmark/prepare_inputs.py \
  --input      benchmark/output/sample_negatives_output/balanced_collapsed.csv \
  --output-dir benchmark/prepared_inputs/cel_balanced \
  --prefix     cel_balanced \
  --truth-bed  benchmark/download/data/cel-precursors-no-v2.bed

# Step 3: run all tools (unified output, required for evaluation)
for TOOL in mire2e mirdnn dnnpremir deepmir deepmirgene mustard; do
  python3 tools/inference.py \
    --tool        $TOOL \
    --output-name cel_balanced \
    --config      benchmark/balanced_benchmark/configs/cel_balanced/${TOOL}.yaml
done

# Step 4: evaluate
python3 benchmark/balanced_benchmark/evaluate_outputs.py \
  --prepared-dir benchmark/prepared_inputs/cel_balanced \
  --results-dir  results \
  --output-dir   benchmark/evaluated/cel_balanced \
  --prefix       cel_balanced \
  --tools        all

# Step 5: plot
python3 benchmark/balanced_benchmark/plot_metrics.py \
  --metrics-csv benchmark/evaluated/cel_balanced/metrics.csv \
  --curves-json benchmark/evaluated/cel_balanced/curves.json \
  --out-dir     benchmark/evaluated/cel_balanced \
  --title       "C. elegans Pre-miRNA Benchmark (Balanced)"
```

Results will be in `benchmark/evaluated/cel_balanced/`.

---

## File Structure

```
pre-miRBench/
├── tools/
│   ├── inference.py                   # Orchestrator — runs any tool via Docker
│   └── {tool}/
│       ├── Dockerfile
│       ├── inference.py               # Tool-specific script (run inside Docker)
│       ├── patch_upstream.py          # Build-time source patches (dnnpremir, deepmir, deepmirgene)
│       └── runtime_predictor.py       # Runtime-swapped script (deepmir only)
├── benchmark/
│   ├── balanced_benchmark/
│   │   ├── prepare_inputs.py          # Step 2 — generate per-tool FASTA/BED
│   │   ├── evaluate_outputs.py        # Step 5 — compute metrics
│   │   ├── plot_metrics.py            # Step 6 — generate figures
│   │   ├── tool_adapters.py           # Shared: sequence cropping, ID mapping
│   │   ├── metrics.py                 # ROC AUC, PR AUC (sklearn)
│   │   └── configs/
│   │       └── cel_balanced/          # Per-tool YAML configs for C. elegans
│   ├── output/
│   │   ├── fold_output/               # RNAfold windows (C. elegans)
│   │   └── sample_negatives_output/   # Balanced dataset CSV
│   └── download/data/                 # ce11.fa, cel-precursors-no-v2.bed
├── results/
│   └── {tool}/{output-name}/
│       └── predictions.csv            # Unified output (--norm-output y)
└── benchmark/evaluated/
    └── {output-name}/
        ├── metrics.csv
        ├── curves.json
        └── *.png
```

---

## Notes and Troubleshooting

- **Run everything from the repo root.** Using relative paths from a subdirectory will cause file-not-found errors.
- **Docker must be running** before any inference command. Start Docker Desktop if you see a "cannot connect to Docker daemon" error.
- **Re-running a tool** with the same `--output-name` overwrites its `predictions.csv`. Use a different `--output-name` to keep multiple runs side by side.
- **MuStARD without conservation** (`MuStARD-mirSF`) scores near 0.5 AUC on cross-species data — this is expected. To use conservation, place per-chromosome PhyloP files (`{chrom}.wigFix.gz`) in `consDir` and switch to `model: MuStARD-mirSFC-U` with `inputMode: sequence,RNAfold,conservation`.
- **`--norm-output n` skips the evaluate/plot steps.** Evaluation requires the unified `predictions.csv` produced by the default `--norm-output y`.
- **Image rebuild after code changes:** modifying `tools/mire2e/inference.py`, `tools/mirdnn/inference.py`, `tools/dnnpremir/inference.py`, `tools/deepmirgene/inference.py`, or `tools/deepmir/runtime_predictor.py` takes effect immediately — those files are volume-mounted into the container at runtime. Changes to `Dockerfile` or `patch_upstream.py` require a full image rebuild.
