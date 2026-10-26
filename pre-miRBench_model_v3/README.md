# pre-miRBench model v3

This directory contains standalone training, inference, and evaluation programs
for the pre-miRBench v3 model. The trained model files are not stored in the
repository.

The model was developed with Agentomics on the fixed `premirbench_v3` training
and validation partitions. Agentomics ran 20 iterations and selected
iteration 19. The model averages the positive-class probabilities of three
convolutional networks:

1. aligned network 1 (`aligned_network_1.pt`, class `AlignedPairCNN`, seed 42,
   trained in iteration 19);
2. aligned network 2 (`aligned_network_2.pt`, the same class and training
   recipe, seed 73, trained in iteration 18);
3. a folded-grid network (`folded_grid_network.pt`, class
   `FoldedGridPreMiRNACNN`, seed 17, trained in iteration 11).

All networks receive the 200-nt RNA sequence, its RNAfold dot-bracket
structure, and its minimum free energy (MFE); species and IDs are not inputs.
Each window is encoded as 13 values per position: the nucleotide (A, C, G, U),
the dot-bracket symbol, the number of open base pairs after the position
divided by 100, the signed distance to the pairing partner divided by 199, and
the partner's nucleotide. The MFE is standardized with the training mean and
standard deviation, after replacing missing values with the training median.

- The aligned networks place the 200-nt window in a 240-position frame so that
  the terminal loop nearest the window centre lies at the same position of the
  frame; a window without a terminal loop centred within 20 nt of the window
  centre is placed in the middle of the frame. The shift is computed from the
  structure only, and a 14th channel marks the positions that hold the window. A pair-message layer adds
  the hidden state of each paired base's partner to that base.
- The folded-grid network has the same one-dimensional encoder on the unshifted
  window and a second, two-dimensional encoder over a 100-by-100 grid that
  folds the window at its centre: rows are the first 100 positions, columns the
  last 100 in reverse order, and the grid holds both nucleotides and a channel
  marking the base pairs of the structure.

The positive-class probability is the float32 mean of the three sigmoid
probabilities, stored as float64; the prediction is 1 when it is at least 0.5.

The feature, representation, and architecture code in `model_code/` is copied
unchanged from the Agentomics iteration-19 deployment. `provenance.json` lists
its source paths and SHA-256 hashes.

## Reported test results

Results of the original iteration-19 model on the four `premirbench_v3` tests
(`evaluate.py`, CPU). AP is average precision; the 95% interval comes from
10,000 bootstrap resamples of each test with positives and negatives resampled
separately (seed 42). Precision, recall and F1 are at the threshold that
maximizes F1 on the validation partition (0.4045). Validation AP is 0.9740.

| test | positives | negatives | AP (95% interval) | AUROC | precision | recall | F1 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1: known species, known families | 873 | 8,730 | 0.966 (0.956-0.975) | 0.988 | 0.950 | 0.921 | 0.935 |
| 2: known species, held-out families | 890 | 8,900 | 0.874 (0.855-0.891) | 0.975 | 0.896 | 0.656 | 0.757 |
| 3: held-out species, known families | 345 | 3,450 | 0.982 (0.972-0.991) | 0.995 | 0.953 | 0.939 | 0.946 |
| 4: held-out species, held-out families | 167 | 1,670 | 0.838 (0.789-0.884) | 0.947 | 0.955 | 0.503 | 0.659 |

Retraining with `train.py` and the default seeds on an NVIDIA A10, the GPU of
the Agentomics run, rebuilt aligned network 1 and the folded-grid network
bitwise identical to the original files. Aligned network 2 differed from the
first epoch on (validation AP 0.9658 against 0.9685, the same selected epoch
29). The retrained model has validation AP 0.9722 and test AP 0.968, 0.872,
0.983 and 0.828. The original iteration-18 script, run on the same GPU,
rebuilds aligned network 2 bitwise identical to the original; `train.py`
trains the three networks one after the other in one process, while the
Agentomics runs trained each network in its own process. The cause of the
difference has not been identified. A retrain on an NVIDIA RTX A4000 gave
validation AP 0.9751 and test AP 0.967, 0.870, 0.979 and 0.828.

## Environment

From this directory:

```bash
uv sync --locked
```

The lock file uses Python 3.12 and the package versions of the Agentomics model
environment, including PyTorch 2.9.0 (CUDA 12.8 build on Linux), NumPy 2.5.4,
pandas 2.3.1, and scikit-learn 1.7.1.

## Model bundle

`train.py` writes a model bundle directory with the three network files, the
fitted preprocessing (`preprocessing.json`, the MFE median, mean and standard
deviation of the training split), a copy of `model_code/`, and
`deployment_manifest.json` with the seeds, the selected epochs and the SHA-256
hashes of all these files.

The original iteration-19 model can be packaged into the same layout from the
Agentomics run directory:

```bash
uv run python package_original.py \
  --source /path/to/premirbench_v3_run1/best_iteration_snapshot/model_training/training_artifacts \
  --artifacts-dir original_artifacts
```

The script checks every copied file against the hashes in the agent's
`model_config.json` and the checkpoints, preprocessing and `model_config.json`
also against `provenance.json`, and renames the agent's `component_seed_42.pt`,
`component_seed_73.pt` and `component_seed_17.pt` to the three network files
above.

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
`--device cpu` for CPU inference.

## Evaluation

The dataset directory must contain the four test split directories, each with
`input/samples.tsv` and `labels.csv`:

```bash
uv run python evaluate.py \
  --dataset-dir /path/to/premirbench_v3 \
  --artifacts-dir original_artifacts \
  --output-dir evaluation_reproduced
```

The command regenerates record-level predictions, per-test JSON metrics, and a
combined `metrics.csv`. Both `label` and `numeric_label` columns are accepted.

## Training

Retrain the model using only the training and validation partitions:

```bash
uv run python train.py \
  --train-data /path/to/premirbench_v3/train \
  --validation-data /path/to/premirbench_v3/validation \
  --artifacts-dir retrained_artifacts \
  --seeds 42 73 17
```

The script replays the three Agentomics training runs behind the iteration-19
model, one after the other, with the settings of the original scripts. The
preprocessing is fitted once on the training split. Each network is trained
from scratch with the same recipe:

| setting | value |
| --- | --- |
| optimizer | AdamW, learning rate 0.0005, weight decay 0.001 |
| batches | 256 training windows, shuffled every epoch |
| loss | binary cross-entropy on the network output |
| augmentation | mixup within each batch of the hidden states after the pair-message layer (and, in the folded-grid network, the grid encoding), the MFE and the labels, with weights from Beta(0.2, 0.2) |
| gradient clipping | norm 5 |
| learning rate schedule | halved when the validation AP has not risen by 0.01% for more than 3 epochs, down to 0.00001 |
| checkpoint | the epoch with the highest validation AP |
| early stopping | after 15 epochs without a validation AP gain of at least 0.0001, at most 100 epochs |

`--seeds` gives the seeds of aligned network 1, aligned network 2, and the
folded-grid network; the default 42 73 17 uses the original seeds. A training
split with fewer than 1,000 rows runs 2 epochs, as the smoke check of the
Agentomics scripts. The script writes the model bundle and
`training_history.json` with the seeds, the selected epochs, and the validation
AP of every epoch, and checks that the packaged inference reproduces the
validation predictions of the trained networks exactly. It refuses to overwrite
a nonempty output directory.

Training runs on one CUDA GPU, as in the Agentomics runs (`--device cuda`, the
default); `--device cpu` exists for smoke tests. On CPU, a 2-epoch run on a
600-window training subset gives networks bitwise identical to those of the
original iteration-11, -18 and -19 scripts. On a GPU, retraining with the
original seeds follows the original steps and random-number use but is not
expected to be bitwise identical across GPUs, drivers, and library builds.

## Files

```text
model_code/           unchanged Agentomics feature, representation, and architecture code
train.py              fixed retraining workflow
inference.py          checksum-verified standalone inference
evaluate.py           four-test prediction and metric workflow
package_original.py   packages the original iteration-19 model files as a bundle
provenance.json       Agentomics run, dataset, training recipe, and source provenance
```
