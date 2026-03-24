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

WORKDIR /opt/dnnpremir

COPY environment.yml .

RUN mamba env create \
    -f environment.yml && \
    conda clean -afy

ENV PATH="/opt/conda/envs/dnnpremir/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/envs/dnnpremir/lib"

RUN git clone https://github.com/zhengxueming/dnnPreMiR.git dnnpremir_src

COPY inference.py ./

WORKDIR /work

ENTRYPOINT ["python", "/opt/dnnpremir/inference.py"]
