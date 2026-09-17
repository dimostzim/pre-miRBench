# pre-miRBench model

This directory contains the released pre-miRBench model, its trained weights,
record-level test predictions, and standalone training, inference, and
evaluation programs.

The model was developed with Agentomics and selected using the fixed training
and validation partitions. It is an equal-weight probability ensemble of three
neural-network checkpoints:

1. one PairMessageCNN trained with reverse-complement augmentation and evaluated
   by averaging the original and reverse-complement views;
2. one species-conditioned graph-context bidirectional GRU trained and evaluated
   with the same two sequence views; and
3. one species-conditioned graph-context bidirectional GRU trained and evaluated
   on the original view.

All components receive the 200-nt RNA sequence, RNAfold dot-bracket structure,
and minimum free energy. The two graph-GRU components also receive species and
13 summary features derived from sequence and structure. The final probability
is the arithmetic mean of the three component probabilities. The released
checkpoint contains 1,178,309 trainable parameters across its three components.

## Reported test results

Average precision (AP; area under the precision-recall curve) is the primary
metric.

| test set | records | AP |
| --- | ---: | ---: |
| known species, known family | 7,777 | 0.986943 |
| known species, held-out family | 7,447 | 0.981334 |
| held-out species, known family | 2,277 | 0.989523 |
| held-out species, held-out family | 759 | 0.965927 |
| mean | 18,260 total | 0.980932 |

The model ranked first on the first three tests and second on the joint species
and family holdout. Its mean AP was 0.009046 higher than the strongest mean AP
among the six retrained published predictors.

## Environment

From this directory:

```bash
uv sync --locked
```

The lock file uses Python 3.12 and the package versions from the completed
Agentomics run, including PyTorch 2.9.0, NumPy 2.5.1, pandas 2.3.1, and
scikit-learn 1.7.1.

## Inference

The input directory must contain `samples.tsv` with these columns in order:

```text
id  species  sequence_rna  structure  mfe
```

Run inference with the released checkpoint:

```bash
uv run python inference.py \
  --input /path/to/split/input \
  --output predictions.csv
```

The output has one row per input ID in the same order and the columns `id`,
`prediction`, `probability_0`, and `probability_1`. SHA-256 hashes for the
checkpoint, architecture code, and fitted preprocessing metadata are verified
against `training_artifacts/deployment_manifest.json` before inference.

GPU execution is selected automatically when CUDA is available. Use
`--device cpu` for CPU inference. Data-loader workers default to zero so the
program also works in containers with limited shared memory.

## Evaluation

The dataset directory must contain the four test split directories, each with
`input/samples.tsv` and `labels.csv`:

```bash
uv run python evaluate.py \
  --dataset-dir /path/to/premirbench_mirgenedb71 \
  --output-dir evaluation_reproduced
```

The command regenerates record-level predictions, per-test JSON metrics, and a
combined `metrics.csv`. Both `label` and `numeric_label` columns are accepted.

The committed `evaluation/` directory contains the predictions from the
reported run. Its `*.metrics.json` files contain the complete metrics used by
the manuscript, including positive-class and macro-averaged F1.

## Training

Retrain the fixed architecture using only the public training and validation
partitions:

```bash
uv run python train.py \
  --train-data /path/to/premirbench_mirgenedb71/train \
  --validation-data /path/to/premirbench_mirgenedb71/validation \
  --artifacts-dir retrained_artifacts
```

The script fits all preprocessing statistics on the training split, trains the
three components, selects each checkpoint by validation AP, and writes a
manifest-compatible artifact bundle. It refuses to overwrite a nonempty output
directory.

The released weights reproduce the manuscript predictions. Retraining follows
the selected architecture and training procedure but is not expected to be
bitwise identical across GPUs and PyTorch kernels.

## Files

```text
training_artifacts/  weights, manifest, fitted metadata, and architecture code
evaluation/          reported predictions, labels, and metrics for four tests
train.py             fixed three-component retraining workflow
inference.py         checksum-verified standalone inference
evaluate.py          four-test prediction and metric workflow
provenance.json      Agentomics run, dataset, and result provenance
checksums.sha256     SHA-256 checksums for released artifacts and predictions
```
