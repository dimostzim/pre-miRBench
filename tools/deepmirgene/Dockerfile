FROM continuumio/miniconda3:23.5.2-0

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

WORKDIR /opt/deepmirgene

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential wget libc6-dev libcrypt-dev && \
    rm -rf /var/lib/apt/lists/*

COPY environment.yml .

RUN conda install -n base -c conda-forge mamba -y && \
    mamba env create -f environment.yml && \
    conda clean -afy

# create symlink so conda python can find crypt.h
RUN ln -sf /usr/include/crypt.h /opt/conda/envs/deepmirgene/include/python3.7m/crypt.h

ENV PATH="/opt/conda/envs/deepmirgene/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/envs/deepmirgene/lib"
ENV KERAS_BACKEND="theano"
ENV THEANO_FLAGS="optimizer=None,linker=py"

RUN git clone https://github.com/eleventh83/deepMiRGene.git deepmirgene_src

COPY inference.py ./

# build ViennaRNA from source with Python bindings (save logs to build context)
RUN tmpdir=$(mktemp -d) && \
    VR_URL="https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_4_x/ViennaRNA-2.4.18.tar.gz" && \
    wget -O "$tmpdir/ViennaRNA.tar.gz" "$VR_URL" && \
    tar -xzf "$tmpdir/ViennaRNA.tar.gz" -C "$tmpdir" && \
    cd "$tmpdir/ViennaRNA-2.4.18" && \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    LDFLAGS="-Wl,-rpath,/opt/conda/envs/deepmirgene/lib -L/opt/conda/envs/deepmirgene/lib" \
    ./configure --with-python3=/opt/conda/envs/deepmirgene/bin/python --without-perl --without-forester --disable-doc --prefix=/opt/conda/envs/deepmirgene 2>&1 | tee /opt/deepmirgene/viennarna_configure.log && \
    make -j"$(nproc)" 2>&1 | tee /opt/deepmirgene/viennarna_make.log && \
    make install 2>&1 | tee /opt/deepmirgene/viennarna_install.log && \
    rm -rf "$tmpdir"

# test that Theano can be imported successfully
RUN /opt/conda/envs/deepmirgene/bin/python -c "import theano; print('Theano loaded successfully')"

WORKDIR /work

ENTRYPOINT ["python", "/opt/deepmirgene/inference.py"]
