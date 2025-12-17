# DeepMir

## Setup

### Option 1: Local Conda Environment
Installs DeepMir dependencies and clones source repository.

Install conda: https://docs.anaconda.com/miniconda/install/

Then run:

```bash
./setup.sh
```

### Option 2: Docker

Install Docker Desktop: https://www.docker.com/products/docker-desktop/

Ensure Docker is running before building image or running containers.

Then run:

```bash
docker build -t deepmir:latest .
```

## Download Test Data

```bash
./download_test_data.sh
```

## Run Inference

### Option 1: Local Conda Environment

Activate the environment:
```bash
conda activate deepmir
```

Run inference on test data:
```bash
python inference.py \
  --input data/test.fa \
  --output results \
  --model fine_tuned_cnn
```

### Option 2: Docker

```bash
docker run --rm --platform linux/amd64 \
  -v $(pwd)/data:/work/data \
  -v $(pwd)/results:/work/results \
  deepmir:latest \
  --input /work/data/test.fa \
  --output /work/results \
  --model fine_tuned_cnn
```

## Parameters

**required:**
- `--input`: FASTA file with RNA sequences.

**default:**
- `--output` (default: `"results"`): Output directory.
- `--model` (default: `"fine_tuned_cnn"`): Pre-trained model (`fine_tuned_cnn` or `base_cnn`).

## Models

| Model | Type | Training Method | Performance |
|-------|------|-----------------|-------------|
| **fine_tuned_cnn** (default) | VGG-based CNN | Fine-tuned | **Best performance** |
| base_cnn | VGG-based CNN | Base training | Good |

The **fine_tuned_cnn** model is recommended.

## Method

DeepMir uses a unique approach:
1. **Hairpin Image Generation**: Converts RNA sequences to 2D images using a Java-based hairpin structure visualizer
2. **Image Processing**: Processes hairpin images (25×100 pixels, RGB)
3. **CNN Classification**: Uses VGG-based CNN to classify images as pre-miRNA or not

## Output

The tool generates:
- `results.csv`: Predictions with columns (hairpin, sequence, fold, label)
- `images/`: Directory with hairpin structure images (.png files)
- `images.npz`: Numpy array of image data
- `names.npz`: Numpy array of sequence names
