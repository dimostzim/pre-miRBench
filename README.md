# pre-miRBench

pre-miRBench is a multispecies benchmark for precursor-miRNA prediction built from MirGeneDB. It includes the dataset construction pipeline, benchmarking code for published predictors, and the released pre-miRBench model.

The canonical benchmark currently uses 71 species, 200 nt sequence windows, and a 1:10 positive:negative ratio, with separate tests for species and miRNA-family generalization.

## Released pre-miRBench model

The released model is in [`pre-miRBench_model/`](pre-miRBench_model/).

It includes trained weights, standalone inference, retraining and evaluation code, provenance, checksums, and the reported predictions/metrics. The model is a three-component neural-network ensemble using sequence, RNAfold structure and minimum free energy, with species/context features in two components.

See [`pre-miRBench_model/README.md`](pre-miRBench_model/README.md) for inference and reproduction instructions.

## Build the dataset

Create the environment:

```bash
conda env create -f pipeline/environment.yml
conda activate premirbench
```

The environment contains the dependencies needed for dataset construction, including ViennaRNA (`RNAfold`), `bedtools`, NumPy and scikit-learn.

Download the MirGeneDB/genome source data:

```bash
bash pipeline/download_data.sh
```

Build the canonical dataset:

```bash
python pipeline/build.py
```

The default output is:

```text
data/datasets/mirgenedb_71/
```

Useful options:

```bash
python pipeline/build.py --jobs 12
python pipeline/build.py --resume
python pipeline/build.py --data-dir /path/to/large/disk/premirbench-data
```

The source genomes are large; the combined reference genome is approximately 92 GB.

## Benchmark design

The dataset is split into training, validation, and four test conditions:

| test | species | miRNA family |
| --- | --- | --- |
| known species / known family | seen | seen |
| known species / held-out family | seen | unseen |
| held-out species / known family | unseen | seen |
| held-out species / held-out family | unseen | unseen |

Exact prepared 100 nt sequences are globally deduplicated across splits and labels to prevent sequence leakage.

## Published predictor benchmarking

The repository also contains Dockerized training/inference adapters for:

`DeepMir`, `DeepMirGene`, `DNNpreMiR`, `miRDeepNN`, `miRe2e`, and `MuStARD`.

The main entry points are:

```text
pipeline/train.py
pipeline/evaluate.py
tools/
```

Docker is only required for these published predictor containers; it is not required to build the pre-miRBench dataset.

## Repository structure

```text
pre-miRBench_model/          released pre-miRBench model and reproducibility files
panels/mirgenedb_71/         species panel, source URLs and provenance
pipeline/download_data.sh   download and validate source data
pipeline/build.py            canonical dataset build command
pipeline/build_dataset.py    lower-level dataset builder
pipeline/train.py            published-tool training wrapper
pipeline/evaluate.py         benchmarking and evaluation
tools/                       Dockerized published predictors
```
