# pre-miRBench model v2

This directory contains standalone training, inference, and evaluation programs
for the pre-miRBench v2 model. The trained model files are not stored in the
repository.

The model was developed with Agentomics on the fixed `premirbench_v2` training
and validation partitions. Agentomics ran 20 iterations and selected
iteration 19. The model is a fixed probability blend of three classifiers:

1. one XGBoost model with 16,721 depth-3 trees on 16,354 tree features:
   13,357 sequence, structure, and positional features followed by 2,997
   features aligned to the three terminal loops closest to the window centre;
2. one neural network (`frozen_neural.pt`) with a convolutional branch over an
   18-by-200 positional tensor of nucleotides, dot-bracket symbols, and base
   pairs, and a fully connected branch over 1,837 global features; and
3. a second neural network (`new_neural.pt`) with the same architecture and
   training recipe and a different seed.

All components receive the 200-nt RNA sequence, RNAfold dot-bracket structure,
and minimum free energy; species and IDs are not inputs. The positive-class
probability is computed in float64 as
`0.80 * p_xgboost + 0.10 * p_first_network + 0.10 * p_second_network`, and the
prediction is 1 when this probability is at least 0.5. Each neural network has
1,249,153 trainable parameters. The file names `frozen_neural.pt` and
`new_neural.pt` are the names used by Agentomics for the first and second
neural network.

The feature, representation, and architecture code in `model_code/` is copied
unchanged from the Agentomics iteration-19 deployment. `provenance.json` lists
its source paths and SHA-256 hashes.

## Reported test results

To be added.

## Environment

From this directory:

```bash
uv sync --locked
```

The lock file uses Python 3.12 and the package versions of the Agentomics model
environment, including PyTorch 2.9.0 (CUDA 12.8 build on Linux), XGBoost 3.4.1,
NumPy 2.5.4, pandas 2.3.1, and scikit-learn 1.7.1.

## Model bundle

`train.py` writes a model bundle directory with the three model files
(`model.ubj`, `frozen_neural.pt`, `new_neural.pt`), the fitted preprocessing
(`augmented_tree_imputer.joblib`, `neural_bundle/global_preprocessor.joblib`,
and `tree_bundle/imputer.joblib`), a copy of `model_code/`, and
`deployment_manifest.json` with the SHA-256 hashes of all these files. The
Agentomics representation code loads and checks `tree_bundle/imputer.joblib`,
although only the helper XGBoost used it during training.

The original iteration-19 model can be packaged into the same layout from the
Agentomics run directory:

```bash
uv run python package_original.py \
  --source /path/to/premirbench_v2_run2/best_iteration_snapshot/model_training/training_artifacts \
  --artifacts-dir original_artifacts
```

The script checks every copied file against the Agentomics `manifest.json`.

## Inference

The input directory must contain `samples.tsv` with the columns `id`,
`sequence_rna`, `structure`, and `mfe` (any order; other columns such as
`species` are ignored). Sequences must be 200 nt with a balanced 200-character
dot-bracket structure; folding is not part of inference.

Run inference with a model bundle:

```bash
uv run python inference.py \
  --input /path/to/split/input \
  --output predictions.csv \
  --artifacts-dir original_artifacts
```

The output has one row per input ID in the same order and the columns `id`,
`prediction`, `probability_0`, and `probability_1`. SHA-256 hashes of all
bundle files are verified against `deployment_manifest.json` before inference.
On CPU, the output for the original iteration-19 bundle is byte-identical to the
output of the Agentomics inference script.

GPU execution is selected automatically when CUDA is available. Use
`--device cpu` for CPU inference. CPU and GPU probabilities differ by at most
about 4e-6 on the validation split.

## Evaluation

The dataset directory must contain the four test split directories, each with
`input/samples.tsv` and `labels.csv`:

```bash
uv run python evaluate.py \
  --dataset-dir /path/to/premirbench_v2 \
  --artifacts-dir original_artifacts \
  --output-dir evaluation_reproduced
```

The command regenerates record-level predictions, per-test JSON metrics, and a
combined `metrics.csv`. Both `label` and `numeric_label` columns are accepted.

## Training

Retrain the model using only the training and validation partitions:

```bash
uv run python train.py \
  --train-data /path/to/premirbench_v2/train \
  --validation-data /path/to/premirbench_v2/validation \
  --artifacts-dir retrained_artifacts \
  --seed 42
```

The script replays the Agentomics training steps behind the iteration-19 model
with the settings that the original run selected. It extracts the features of
each split once and fits all preprocessing on the training split. The
validation split selects the epochs of both neural networks and the number of
XGBoost trees:

| step | Agentomics source | selection on the validation split |
| --- | --- | --- |
| helper XGBoost on the 13,357 tree features | iteration 10 | early stopping after 400 rounds without improvement of the XGBoost `aucpr` metric |
| first neural network | iteration 13 | epoch with the best AP of 0.8 helper XGBoost + 0.2 first network |
| second neural network | iteration 14 | epoch with the best AP of 0.8 helper XGBoost + 0.1 first + 0.1 second network |
| final XGBoost on the 16,354 tree features | iteration 18 | early stopping after 600 rounds without improvement of the blend AP |
| continuation of the final XGBoost at learning rate 0.01 | iteration 19 | total tree count with the best blend AP; stops after 1,000 rounds without improvement |

The helper XGBoost is not part of the deployed model. `--seed` seeds the helper
XGBoost, the first neural network, and both stages of the final XGBoost; the
second neural network uses `--seed + 1983`. The default seed 42 therefore uses
the original seeds 42 and 2025. The script writes the model bundle and
`training_history.json` with the seeds, selected epochs and tree counts, and
validation APs. It refuses to overwrite a nonempty output directory.

Training runs on one CUDA GPU, as in the Agentomics run (`--device cuda`, the
default); `--device cpu` exists for smoke tests and drops fp16 autocast. Retraining with
seed 42 follows the original steps and random-number use but is not expected to
be bitwise identical across GPUs, drivers, and library builds.

## Files

```text
model_code/           unchanged Agentomics feature, representation, and architecture code
train.py              fixed retraining workflow
inference.py          checksum-verified standalone inference
evaluate.py           four-test prediction and metric workflow
package_original.py   packages the original iteration-19 model files as a bundle
provenance.json       Agentomics run, dataset, training recipe, and source provenance
```
