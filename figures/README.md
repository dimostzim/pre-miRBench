# Manuscript figures

`figure1_split_matrix.svg` and `figure1_split_matrix.png` show the
pre-miRBench development partitions separately from the 2 × 2 test design.
All six partitions contain distinct records. Species and miRNA-family status
are defined relative to the training partition.

![pre-miRBench split design](figure1_split_matrix.png)

Test 2 and Test 4 each contain 59 held-out families. Fifteen families occur in
both tests, while 44 are unique to each test; their union therefore contains
all 103 families absent from training.

Recreate the figure with:

```bash
python figures/make_figure1_split_matrix.py
```
