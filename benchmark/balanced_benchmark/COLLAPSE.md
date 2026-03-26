# Collapse Mechanism

This repository generates two positive-window views from the folding output:

- `positives.csv`
- `positives_collapsed.csv`

The difference is:

- `positives.csv` keeps every folded window that fully contains a known pre-miRNA
- `positives_collapsed.csv` keeps exactly one representative window per miRNA

## Why Collapse Exists

Sliding windows overlap heavily. A single pre-miRNA can be fully contained by several nearby 200 nt windows. If all of those windows are kept as positives, the same biological event is counted multiple times.

The collapsed view removes that redundancy by selecting one window per miRNA.

## Exact Selection Rule

For each miRNA in the BED file:

1. Find all windows on the same chromosome that fully contain the miRNA.
2. Compute the miRNA midpoint:
   - `(start + end) / 2`
3. Compute each candidate window midpoint:
   - `(window_start + window_end) / 2`
4. Rank candidate windows by:
   - smallest absolute distance between window midpoint and miRNA midpoint
   - then lower MFE
   - then earlier genomic start
5. Keep the best-ranked window only.

This is implemented in:

- [`find_mirna_windows.py`](/home/dtzim01/pre-miRBench/benchmark/fold/find_mirna_windows.py)

The ranking key in code is effectively:

```python
(distance_to_mirna_midpoint, mfe, window_start)
```

Because MFE is more negative for stronger folding, "lower MFE" means a more negative value wins ties.

## Output Semantics

### `positives.csv`

- one row per positive window
- a single miRNA may appear in multiple rows
- columns include:
  - `contained_mirnas`
  - `num_mirnas`

### `positives_collapsed.csv`

- one row per target miRNA
- adds:
  - `target_mirna`
- still includes:
  - `contained_mirnas`
  - `num_mirnas`

This means the chosen representative window for one miRNA may still contain more than one miRNA.

## Relationship To The Four Benchmark Datasets

The four downstream datasets are:

- `balanced.csv`
- `imbalanced.csv`
- `balanced_collapsed.csv`
- `imbalanced_collapsed.csv`

They use positives as follows:

- `balanced.csv` and `imbalanced.csv` use `positives.csv`
- `balanced_collapsed.csv` and `imbalanced_collapsed.csv` use `positives_collapsed.csv`

So the collapsed datasets are the "one window per miRNA" benchmark variants.
