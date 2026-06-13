# pre-miRBench

Pipeline for retraining and evaluating pre-miRNA prediction tools.

## Environment

The dataset builder needs `RNAfold`, which is provided by ViennaRNA in the
conda environment:

```bash
conda env create -f pipeline/environment.yml
conda activate premirbench
```

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

Example for a 1:10 positive:negative ratio on a 96-CPU machine:

```bash
python pipeline/build_dataset.py \
  --ratio 10 \
  --cpus 8 \
  --species-jobs 12
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

## Dataset Options

`pipeline/build_dataset.py` builds `data/datasets/diverse20` from the downloaded
species panel. `--ratio` is the negative:positive ratio, so `--ratio 10` means
ten negative windows for each positive precursor.

| Option | Default | Meaning |
| --- | --- | --- |
| `--panel` | `data/raw/diverse20/panel.tsv` | Species manifest from `pipeline/download_data.sh`. |
| `--output-dir` | `data/datasets/diverse20` | Final dataset, combined genome, split summary, and per-tool inputs. |
| `--work-dir` | `data/work/build_dataset` | Intermediate per-species files. |
| `--species` | all auto species | Comma-separated species codes to include, for example `hsa,mmu,dre`. |
| `--heldout-species` | `dre,dme` | Comma-separated species held out for the species test split. |
| `--ratio` | `5` | Negative:positive ratio. Use `10` for 1:10. |
| `--window` | `200` | Sequence window length around each precursor. |
| `--step` | `50` | Sliding-window step used when scanning candidate negatives. |
| `--max-negative-windows-per-species` | `50000` | Cap on candidate negative windows per species before mining. |
| `--sequential-negative-scan` | off | Scan negative windows in FASTA order instead of balanced chromosome sampling. |
| `--max-repeat-frac` | `0.1` | Drop windows with more than this fraction of repeat-masked bases. |
| `--min-mfe` | `-10.0` | Minimum RNAfold MFE threshold for candidate hairpins. |
| `--min-paired-frac` | `0.40` | Minimum paired-base fraction for candidate hairpins. |
| `--min-stem` | `8` | Minimum stem length for candidate hairpins. |
| `--max-loop` | `25` | Maximum loop length for candidate hairpins. |
| `--cpus` | `8` | RNAfold jobs per species. |
| `--species-jobs` | `1` | Number of species processed in parallel. |
| `--mining-rounds` | `4` | Hard-negative mining rounds. |
| `--ensemble-size` | `10` | RandomForest ensemble size used during hard-negative mining. |
| `--trees` | `200` | Trees per RandomForest model during mining. |
| `--consensus` | `0.5` | Consensus threshold for selecting hard negatives. |
| `--mining-jobs` | auto | RandomForest jobs per species. Default is `-1` when sequential, `1` when `--species-jobs > 1`. |
| `--seed` | `42` | Random seed for sampling and splits. |
| `--reuse-existing` | off | Reuse existing intermediate files where supported. |

## Training Options

`pipeline/train.py` trains one Dockerized tool at a time.

| Option | Default | Meaning |
| --- | --- | --- |
| `--tool` | required | Tool to train: `mustard`, `mire2e`, `mirdnn`, `dnnpremir`, `deepmir`, or `deepmirgene`. |
| `--run-name` | required | Output subdirectory under `results/training/<tool>/`. |
| `--dataset-dir` | `data/datasets/diverse20` | Dataset created by `pipeline/build_dataset.py`. |
| `--config` | none | Optional YAML file overriding the derived tool training defaults. |
| `--output-root` | `results/training` | Root directory for training outputs. Supports env vars such as `$SCRATCH`. |

Common `--config` overrides are `device`, `batch_size`, `epochs`, early-stopping
settings, and tool-specific paths or model parameters. The generated
`inference_config.yaml` is written into each training output directory and is
used later by evaluation.

## Evaluation Options

`pipeline/evaluate.py` evaluates trained tools on held-out splits.

| Option | Default | Meaning |
| --- | --- | --- |
| `--dataset-dir` | `data/datasets/diverse20` | Dataset created by `pipeline/build_dataset.py`. |
| `--training-root` | `results/training` | Root directory containing trained tool outputs. |
| `--run-name` | latest/required by outputs | Training run name to evaluate. |
| `--output-dir` | `results/evaluation/<run-name>` | Evaluation output directory. |
| `--tools` | all tools | Comma-separated tools to evaluate. |
| `--splits` | all held-out splits | Comma-separated splits to evaluate. |
| `--skip-inference` | off | Parse existing raw outputs without rerunning Docker. |
| `--resume` | off | Reuse non-empty raw output directories instead of rerunning inference. |
| `--allow-missing` | off | Skip tools whose trained `inference_config.yaml` is missing. |
| `--dry-run` | off | Print Docker commands without running them. |
| `--log-file` | `<output-dir>/run.log.txt` | Optional evaluation log path. |
