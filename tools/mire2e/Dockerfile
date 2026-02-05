FROM continuumio/miniconda3:latest

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

WORKDIR /opt/mire2e

COPY environment.yml .

RUN conda install -n base -c conda-forge mamba -y && \
    mamba env create -f environment.yml && \
    conda clean -afy

ENV PATH="/opt/conda/envs/mire2e/bin:${PATH}"

RUN python -c "from miRe2e import MiRe2e; MiRe2e(pretrained='hsa'); MiRe2e(pretrained='animals'); print('Models downloaded')"

COPY inference.py ./

WORKDIR /work

ENTRYPOINT ["python", "/opt/mire2e/inference.py"]
