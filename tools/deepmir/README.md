# DeepMir

## Setup

### Option 1: Local Conda Environment
Installs DeepMir dependencies and clones source repository.

**Prerequisites:**
- Conda: https://docs.anaconda.com/miniconda/install/
- Java JRE (for hairpin image generation)

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

**Note:** Docker image includes Java JRE for hairpin image generation.

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

**Note:** Requires Java JRE installed on the system for hairpin image generation.

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

The **fine_tuned_cnn** model is recommended as it uses a fine-tuning procedure during training and achieves the best performance according to the paper.

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

**Note:** The original DeepMir creates output in `user_data/{filename}/`. This wrapper copies all results to the specified output directory for consistency with other tools.
