# this is an old miniconda version with py2.7 support
FROM continuumio/miniconda3:4.12.0

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

WORKDIR /opt/mustard

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY environment.yml .

# normally miniconda supports mamba but this old version does not lol. so we install mamba for faster solving
# because building a linux image on ARM Mac and installing old packages is painfully slow.

# currently we only do cpu version only

RUN conda install -n base -c conda-forge mamba -y && \
    mamba env create -f environment.yml && \
    conda clean -afy
ENV PATH="/opt/conda/envs/mustard/bin:${PATH}"

# mustard source and pretrained models
RUN git clone https://gitlab.com/RBP_Bioinformatics/mustard.git mustard_src
RUN mkdir -p data/models

RUN for model in MuStARD-mirS MuStARD-mirF MuStARD-mirSF MuStARD-mirSC MuStARD-mirFC MuStARD-mirSFC MuStARD-mirSFC-U; do \
      mkdir -p data/models/${model}; \
      wget -O data/models/${model}/CNNonRaw.hdf5 "https://gitlab.com/RBP_Bioinformatics/mustard_paper/-/raw/master/pretrained_models/${model}/CNNonRaw.hdf5"; \
    done

COPY inference.py ./

WORKDIR /work

ENTRYPOINT ["python", "/opt/mustard/inference.py"]
