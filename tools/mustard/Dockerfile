FROM continuumio/miniconda3:23.10.0-1

WORKDIR /opt/mustard

# system deps for downloads and git clone
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# python 2.7 / perl / tf1 env
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy
ENV PATH="/opt/conda/envs/mustard/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=mustard

# mustard source and pretrained assets
RUN git clone https://gitlab.com/RBP_Bioinformatics/mustard.git mustard_src
RUN mkdir -p data/models

RUN for model in MuStARD-mirS MuStARD-mirF MuStARD-mirSF MuStARD-mirSC MuStARD-mirFC MuStARD-mirSFC MuStARD-mirSFC-U; do \
      mkdir -p data/models/${model}; \
      wget -O data/models/${model}/CNNonRaw.hdf5 "https://gitlab.com/RBP_Bioinformatics/mustard_paper/-/raw/master/pretrained_models/${model}/CNNonRaw.hdf5"; \
    done

COPY inference.py ./

WORKDIR /work

ENTRYPOINT ["python", "/opt/mustard/inference.py"]
