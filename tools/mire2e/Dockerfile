FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

WORKDIR /opt/mire2e

COPY environment.yml .

RUN mamba env create \
    -f environment.yml && \
    conda clean -afy

ENV PATH="/opt/conda/envs/mire2e/bin:${PATH}"

RUN pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip install miRe2e

RUN python -c "from miRe2e import MiRe2e; MiRe2e(pretrained='hsa'); MiRe2e(pretrained='animals'); print('Models downloaded')"

COPY inference.py ./

WORKDIR /work

ENTRYPOINT ["python", "/opt/mire2e/inference.py"]
