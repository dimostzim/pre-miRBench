# Benchmarking results

This directory contains the trained model files and record-level predictions
for the six published precursor-miRNA predictors evaluated in pre-miRBench.
Training and evaluation logs are included with the archived outputs.

## Published predictors

| predictor | model file |
| --- | --- |
| DeepMir | `models/deepmir/model.h5` |
| DeepMirGene | `models/deepmirgene/new_test.hdf5` |
| DNNpreMiR | `models/dnnpremir/CNN_model.h5` |
| miR-DNN | `models/mirdnn/model.pmt` |
| miRe2E | `models/mire2e/predictor.pkl`, `structure.pkl`, and `mfe.pkl` |
| MuStARD | `models/mustard/CNNonRaw.hdf5` |

The corresponding container definitions and model adapters remain under
[`tools/`](../tools/). Generated training inputs and preprocessing intermediates
are not included here.

## Predictions and metrics

`predictions/published_tools.csv` contains 109,560 predictions: six predictors
evaluated on each of the four fixed test sets. Every predictor has one score for
each test record, with no missing predictions. The table includes the predictor,
test set, record ID, species, binary label, continuous score, prediction at the
0.5 threshold, and input order.

`metrics.csv` contains the 24 corresponding predictor-test records. Its `auprc`
column preserves the original pipeline field name; the values are average
precision calculated by `pipeline/evaluate.py` and reported as AP in the
manuscript.

## pre-miRBench model

The pre-miRBench implementation, checkpoint, fitted preprocessing metadata, and
retraining and inference programs are maintained in [`model/`](../model/). Its
canonical checkpoint is [`model.pt`](../model/training_artifacts/model.pt), and
its four record-level prediction files are under
[`model/evaluation/`](../model/evaluation/). These files are not duplicated here.

## Logs and integrity

`logs/` preserves the training and evaluation logs from the six-predictor
benchmark run. Verify the archived files from this directory with:

```bash
sha256sum -c checksums.sha256
```
