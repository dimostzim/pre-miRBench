# Training Process

This is the current end-to-end retraining flow for the six pre-miRNA tools:
DeepMir, DeepMirGene, dnnPreMiR, mirDNN, miRe2e, and MuStARD.

## 1. Download raw species data

The diverse panel download script fetches genome FASTA files and MirGeneDB
precursor BED files, normalizes chromosome names, validates BED/genome matches,
and writes a manifest:

```bash
bash benchmark/download/download_diverse20.sh
```

Main output:

- `data/train/raw/diverse20/panel.tsv` - one row per species, including genome
  path, normalized BED path, validation status, and BED row counts.
- `data/train/raw/diverse20/<species>/` - downloaded genome and precursor files.

The dataset builder only uses `panel.tsv` rows with `status=auto`.

## 2. Build the canonical dataset

Use `benchmark/train_data/build_multispecies_dataset.py`:

```bash
python benchmark/train_data/build_multispecies_dataset.py \
  --panel data/train/raw/diverse20/panel.tsv \
  --output-dir data/train/diverse20 \
  --work-dir benchmark/train_data/work_diverse20 \
  --ratio 5 \
  --window 200 \
  --step 50 \
  --heldout-species dre,dme
```

Important behavior:

- Contig names are prefixed as `<species>__<contig>` so records from different
  species cannot collide.
- Positive records are fixed-size windows centered on MirGeneDB `_pre`
  precursor intervals.
- Windows are strand-corrected, converted to RNA alphabet, repeat/N filtered,
  folded with `RNAfold`, and kept only if folding output is valid.
- `--ratio` controls the requested negative:positive ratio for every split.
- Default held-out species are `dre,dme`; these become the external
  `test_species` split.
- For every non-heldout species, the builder tries to reserve one
  positive-bearing chromosome/scaffold for `valid` and one for `test_chrom`.
  A chromosome is eligible only if enough negatives exist on that chromosome for
  the configured ratio.

Main output:

- `data/train/diverse20/dataset.csv` - canonical labeled examples.
- `data/train/diverse20/split_summary.csv` - per-species counts and issues.
- `data/train/diverse20/genome.fa` - combined prefixed genome for MuStARD.
- `data/train/diverse20/tool_inputs/` - per-tool training, validation, and test
  input files.

## 3. Negative generation

Negative generation has two stages.

First, `extract_hairpins.py` scans genomic windows that do not overlap known
MirGeneDB pre-miRNA intervals. It filters out repeat/N-heavy windows, folds both
strands with `RNAfold`, and keeps only hairpin-like windows passing:

- MFE threshold, default `--min-mfe -10.0`
- paired fraction, default `--min-paired-frac 0.40`
- stem length, default `--min-stem 8`
- loop size, default `--max-loop 25`

Second, `mine_negatives.py` mines hard negatives from that hairpin-like pool. It
trains an ensemble of lightweight random forests on positives versus sampled
candidate negatives. Features are MFE, dinucleotide frequencies, dot-bracket
structure features, and sequence entropy. In each mining round, pool windows
that the ensemble repeatedly scores as pre-miRNA-like are added as hard
negatives. Final split selection prefers hard negatives first, then high-scored
remaining pool rows, then random remaining rows if needed.

This gives structurally plausible negatives rather than easy random genomic
background.

## 4. Tool-specific input preparation

`benchmark/train_data/prepare_tool_inputs.py` converts the canonical dataset
into files each tool can consume:

- FASTA tools: DeepMir, DeepMirGene, dnnPreMiR, mirDNN, miRe2e.
- BED/interval tool: MuStARD.

All tools originate from the same canonical 200 nt windows. Tool adapters crop
only when the original architecture expects a shorter input:

- dnnPreMiR: 180 nt
- mirDNN: 160 nt
- miRe2e: 100 nt
- DeepMir, DeepMirGene, MuStARD: full canonical window

Each tool gets:

- `positive` and `negative` train files
- `validation_positive` and `validation_negative`
- `test_chrom_positive` and `test_chrom_negative`
- `test_species_positive` and `test_species_negative`
- `metadata.csv`

## 5. Train models

Training is launched through `tools/train.py`, using the configs in
`configs/train/diverse20/`:

```bash
for tool in deepmir deepmirgene dnnpremir mirdnn mire2e mustard; do
  python tools/train.py \
    --tool "$tool" \
    --run-name diverse20_gpu_1to5 \
    --config "configs/train/diverse20/${tool}_train.yaml" \
    --output-root "$SCRATCH/pre-miRBench/training"
done
```

The wrapper runs Docker with `--gpus all` and writes each trained model plus an
`inference_config.yaml` under:

```text
$SCRATCH/pre-miRBench/training/<tool>/diverse20_gpu_1to5/
```

Current configs use the original tool architectures/parameters where possible,
with explicit validation files and early stopping on validation AUPRC where the
wrapper supports it. MuStARD uses `inputMode: sequence,RNAfold`; its positive
score output is `class_1` for the current `classList: 0,1` mapping.

## 6. Evaluate trained models

Evaluation uses the held-out chromosome and held-out species splits:

```bash
python benchmark/evaluate.py \
  --dataset-dir data/train/diverse20 \
  --training-root "$SCRATCH/pre-miRBench/training" \
  --run-name diverse20_gpu_1to5 \
  --output-dir results/evaluation/diverse20_gpu_1to5
```

Outputs:

- `predictions.csv` - one labeled model score per evaluated record.
- `metrics.csv` - aggregate metrics per tool and split.
- `metrics_by_species.csv` - metrics stratified by species.
- `auprc_by_tool.svg` and `auprc_by_tool.png` - grouped AUPRC plot.
- `run.log.txt` - evaluation and tool console log.

Use `--resume` to reuse existing raw inference outputs when possible, or
`--skip-inference` to only reparse existing raw outputs and regenerate metrics.
