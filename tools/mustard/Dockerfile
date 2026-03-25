FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

WORKDIR /opt/mustard

COPY environment.yml .

RUN mamba env create \
    -f environment.yml && \
    conda clean -afy

ENV PATH="/opt/conda/envs/mustard/bin:${PATH}"

RUN git clone https://gitlab.com/RBP_Bioinformatics/mustard.git mustard_src

# Convert MuStARD's Python helpers from Python 2 syntax so they can run in the
# Python 3 TensorFlow environment used by this image.
RUN 2to3 -w mustard_src/src/lib/python mustard_src/src/utilities/python >/dev/null

# Patch apply_CNN.py: tf.Session -> tf.compat.v1.Session (TF 1.x API removed in TF 2.x)
RUN sed -i 's/tf\.Session(/tf.compat.v1.Session(/g' \
    mustard_src/src/utilities/python/apply_CNN.py

RUN sed -i 's/tf\.ConfigProto(/tf.compat.v1.ConfigProto(/g' \
    mustard_src/src/utilities/python/apply_CNN.py

RUN sed -i "s/gzip.open(input_file,'r')/gzip.open(input_file,'rt')/g" \
    mustard_src/src/lib/python/Files/Format.py

RUN mkdir -p data/models
RUN for model in MuStARD-mirS MuStARD-mirF MuStARD-mirSF MuStARD-mirSC MuStARD-mirFC MuStARD-mirSFC MuStARD-mirSFC-U; do \
      mkdir -p data/models/${model}; \
      wget -O data/models/${model}/CNNonRaw.hdf5 "https://gitlab.com/RBP_Bioinformatics/mustard_paper/-/raw/master/pretrained_models/${model}/CNNonRaw.hdf5"; \
    done

COPY inference.py ./

WORKDIR /work

ENTRYPOINT ["python", "/opt/mustard/inference.py"]
