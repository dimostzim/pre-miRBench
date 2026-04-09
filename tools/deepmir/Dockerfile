FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates git default-jre libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

WORKDIR /opt/deepmir

COPY environment.yml .

RUN mamba env create \
    -f environment.yml && \
    conda clean -afy

ENV PATH="/opt/conda/envs/deepmir/bin:${PATH}"

RUN git clone https://github.com/jacordero/deepmir.git deepmir_src

COPY patch_upstream.py ./
RUN python /opt/deepmir/patch_upstream.py
RUN ln -sf /opt/deepmir/deepmir_src/hairpin_image_generator/bin/Linux/RNAfold \
        /opt/deepmir/deepmir_src/hairpin_image_generator/bin/RNAShapes.exe && \
    chmod +x /opt/deepmir/deepmir_src/hairpin_image_generator/bin/RNAShapes.exe

COPY inference.py ./

WORKDIR /work

ENTRYPOINT ["python", "/opt/deepmir/inference.py"]
