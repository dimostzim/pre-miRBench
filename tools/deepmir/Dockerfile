FROM continuumio/miniconda3:latest

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

WORKDIR /opt/deepmir

# Install Java (required for ImageCalc.jar) and libgomp (needed by RNAfold binary)
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre git libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY environment.yml .

RUN conda env create -f environment.yml && \
    conda clean -afy

ENV PATH="/opt/conda/envs/deepmir/bin:${PATH}"

# Clone deepmir repository (includes pre-trained CNN models and ImageCalc.jar)
RUN git clone https://github.com/jacordero/deepmir.git deepmir_src
# The JAR expects a binary named RNAShapes.exe; reuse the bundled Linux RNAfold.
# Use an absolute symlink target to avoid creating a broken link.
RUN ln -sf /opt/deepmir/deepmir_src/hairpin_image_generator/bin/Linux/RNAfold /opt/deepmir/deepmir_src/hairpin_image_generator/bin/RNAShapes.exe && \
    chmod +x /opt/deepmir/deepmir_src/hairpin_image_generator/bin/RNAShapes.exe

COPY inference.py ./

WORKDIR /work

ENTRYPOINT ["python", "/opt/deepmir/inference.py"]
