FROM continuumio/miniconda3:latest

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

WORKDIR /opt/dnnpremir

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY environment.yml .

RUN conda env create -f environment.yml && \
    conda clean -afy

ENV PATH="/opt/conda/envs/dnnpremir/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/envs/dnnpremir/lib:${LD_LIBRARY_PATH}"

# Clone dnnPreMiR repository (includes pre-trained CNN model)
RUN git clone https://github.com/zhengxueming/dnnPreMiR.git dnnpremir_src

COPY inference.py ./

WORKDIR /work

ENTRYPOINT ["python", "/opt/dnnpremir/inference.py"]
