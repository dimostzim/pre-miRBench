# Reported evaluation artifacts

These files are the complete record-level outputs from the selected model on
the four fixed pre-miRBench tests.

- `eval_predictions_<test>.csv` contains IDs, thresholded predictions, and both
  class probabilities.
- `eval_predictions_<test>.numeric_labels.csv` contains the labels used to
  verify the archived predictions.
- `eval_predictions_<test>.metrics.json` contains AP and the full set of
  threshold-free and 0.5-threshold metrics recalculated from those records.
- `metrics.csv` combines all four recalculated metric records and their
  unweighted mean.

The four prediction files contain 18,260 rows in total. Their ID order matches
the corresponding test input files.
