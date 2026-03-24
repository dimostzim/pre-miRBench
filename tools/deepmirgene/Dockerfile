FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

WORKDIR /opt/deepmirgene

COPY environment.yml .

RUN mamba env create \
    -f environment.yml && \
    conda clean -afy

ENV PATH="/opt/conda/envs/deepmirgene/bin:${PATH}"
ENV KERAS_BACKEND="tensorflow"

RUN git clone https://github.com/eleventh83/deepMiRGene.git deepmirgene_src

COPY inference.py ./

RUN python -c "import tensorflow as tf; print('TensorFlow loaded successfully')"

WORKDIR /work

ENTRYPOINT ["python", "/opt/deepmirgene/inference.py"]
