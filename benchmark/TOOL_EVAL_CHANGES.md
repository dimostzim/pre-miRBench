# Tool Evaluation Changes

This file records the changes made in this repo to enable a unified evaluation of
third‑party pre‑miRNA tools. We do **not** modify upstream tool source code inside
Docker images; instead we add wrappers and evaluation glue around them.

## Orchestration and evaluation

- `benchmark/eval_tools.py` (new): runs tools across datasets, parses outputs, and
  writes a single metrics CSV.
  - Runners: `--runner conda|docker`.
  - Output parsing per tool:
    - mirdnn: `predictions.csv` (score 0–1).
    - dnnpremir: `predictions.txt` (True/False labels only).
    - deepmir: `results.csv` (labels only).
    - deepmirgene: `predictions.txt` (0/1 labels only).
    - mire2e: `predictions.json` (window scores, aggregated to sequence score).
    - mustard: `bed_tracks/*predictions*.bed.gz` (score column).
  - Filters/normalization added for eval stability:
    - `--drop_ambiguous`: drop sequences with non‑ACGUT bases (dnnpremir fails on N).
  - `--deepmir_len N`: center‑trim **only DeepMir inputs** to length N (avoids
      silent drops due to image shape mismatch).
  - `--mustard_win_size N`: center‑trim **only MuStARD BED windows** to length N
      (MuStARD‑mirS expects 100 bp).
  - `--mirdnn_use_fold`: build a fold-format input from `structure`/`mfe`
    columns and **skip RNAfold** (huge speedup on large datasets).
  - GPU usage is **on by default** for supported tools (mirdnn, mire2e) and
    docker runs include `--gpus all`. Use `--no_gpu` to force CPU.
  - GPU support is provided via conda `cudatoolkit`/`cudnn` in the tool
    environments (no CUDA base images). Rebuild images after pulling these
    changes (older images are CPU‑only).
  - `--cpu_tools` lets you force specific tools to CPU (useful if TF/Theano
    throws GPU runtime errors).
  - `--shard_size N` splits large datasets into N‑row shards and runs tools per
    shard, then merges predictions (recommended for imbalanced/full runs).
  - `--shard_root` lets you store all shard files under a shared external
    directory (keeps `benchmark/output/tool_eval/<dataset>` clean).
  - `--reuse_shards` skips re‑sharding if shard CSVs already exist.
  - miRe2e length must remain **100** for the pretrained model; larger values
    will fail with shape errors. Use `--mire2e_step 100` if you want two windows
    per 200‑nt sequence and aggregate with `--mire2e_agg` (default `max`).
  - miRe2e windowing follows tool defaults unless you pass a `tool_args` JSON
    (e.g. `{\"mire2e\": [\"--step\", \"100\"]}`).
  - Progress logging: prints a simple `[dataset/tool]` line per run. Use
    `--verbose` to show tool stdout/stderr.

## Tool wrappers (no upstream code edits)

These scripts normalize input/output paths and formats. They do **not** change
model logic.

- `tools/mirdnn/inference.py`: runs RNAfold + mirdnn eval, writes
  `predictions.csv`.
- `tools/dnnpremir/inference.py`: runs `isPreMiR.py`, writes `predictions.txt`.
- `tools/deepmir/inference.py`: runs DeepMir predictor, copies `results.csv`.
- `tools/deepmirgene/inference.py`: runs `deepMiRGene.py`, writes
  `predictions.txt`.
- `tools/mire2e/inference.py`: runs miRe2e, writes `predictions.json`.
  - Multi-record FASTA support: iterates records and runs miRe2e per record.
  - Optional `--max_records` to cap runtime on very large datasets.
  - When using Docker, `benchmark/eval_tools.py` calls this wrapper explicitly
    (overriding the image entrypoint) so the multi-record logic is used.
- `tools/mustard/inference.py`: runs MuStARD in static mode, writes bed/bedGraph
  predictions under the output dir.
  - `tools/inference.py` (generic wrapper) now runs docker with `--gpus all`.

## Dataset preparation changes that affect evaluation

- `benchmark/fold/find_mirna_windows.py`:
  - Always writes `positives_collapsed.csv` (one window per miRNA).
  - Collapse rule: closest window center to miRNA midpoint; ties broken by lower
    MFE, then earlier start.
  - Adds `target_mirna`, `contained_mirnas`, `num_mirnas` columns to positives.

- `benchmark/make_negative_set/sample_negatives.py`:
  - Produces **four** datasets only:
    - `balanced.csv` (1:1, overlapping)
    - `imbalanced.csv` (all negatives, overlapping)
    - `balanced_collapsed.csv` (1:1, **non‑overlapping** negatives)
    - `imbalanced_collapsed.csv` (all negatives, **non‑overlapping**)
  - Distribution matching (MFE/dinucleotide/structure) is applied only to the
    1:1 datasets.

If you want this document expanded (e.g., exact command lines or version tags),
let me know.
