FROM continuumio/miniconda3:latest

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

WORKDIR /opt/mirdnn

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY environment.yml .

RUN conda install -n base -c conda-forge mamba -y && \
    mamba env create -f environment.yml && \
    conda clean -afy

ENV PATH="/opt/conda/envs/mirdnn/bin:${PATH}"

# Clone mirDNN repository (includes pre-trained models)
RUN git clone --recurse-submodules https://github.com/cyones/mirDNN.git mirdnn_src

COPY inference.py ./

WORKDIR /work

ENTRYPOINT ["python", "/opt/mirdnn/inference.py"]
