# AP uncertainty

`bootstrap_ap.py` calculates record-level, class-stratified bootstrap intervals
for average precision (AP) on each fixed test set. Positive and negative records
are resampled separately with replacement. The same resampled records are used
for all seven models within a test, allowing paired AP differences to be
calculated. The analysis uses 10,000 resamples and seed 42.

Outputs:

- `ap_bootstrap_95ci.csv`: AP and percentile 95% interval for all 28 model-test
  combinations.
- `ap_difference_vs_premirbench_95ci.csv`: paired AP differences between the
  pre-miRBench model and each published model.
- `table2_ap_with_ci.tsv`: the interval estimates formatted for Table 2.

These intervals describe uncertainty from resampling records in the fixed test
sets. They do not include variation from retraining models or selecting other
species.

Run from the repository root:

```bash
python benchmarking_results/uncertainty/bootstrap_ap.py
```
