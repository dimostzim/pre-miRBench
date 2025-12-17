#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$*" == *"--docker"* ]]; then
    image_tag="${IMAGE_TAG:-deepmirgene:latest}"

    BUILD_ARGS=""
    [ -n "$http_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg http_proxy=$http_proxy"
    [ -n "$https_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg https_proxy=$https_proxy"
    [ -n "$HTTP_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTP_PROXY=$HTTP_PROXY"
    [ -n "$HTTPS_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTPS_PROXY=$HTTPS_PROXY"

    docker build --platform linux/amd64 $BUILD_ARGS -t "$image_tag" .
    exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create -f environment.yml

PYTHON_BIN="$(conda info --base)/envs/deepmirgene/bin/python"

# build ViennaRNA from source (python module + libs) if the RNA module is missing
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("RNA") else 1)
PY
then
    CONDA_PREFIX_PATH="$(conda info --base)/envs/deepmirgene"
    JOBS=${JOBS:-$(nproc 2>/dev/null || echo 1)}
    TMPDIR_VR=$(mktemp -d)

    VR_URL="https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_4_x/ViennaRNA-2.4.18.tar.gz"
    wget -O "$TMPDIR_VR/ViennaRNA.tar.gz" "$VR_URL"
    tar -xzf "$TMPDIR_VR/ViennaRNA.tar.gz" -C "$TMPDIR_VR"

    conda run -n deepmirgene bash -lc "\
        set -e; \
        export PATH='$CONDA_PREFIX_PATH/bin':\$PATH; \
        export PYTHON='$PYTHON_BIN'; \
        export PYTHON3='$PYTHON_BIN'; \
        export LD_LIBRARY_PATH='$CONDA_PREFIX_PATH/lib':\${LD_LIBRARY_PATH:-}; \
        cd '$TMPDIR_VR/ViennaRNA-2.4.18' && \
        CC=${CC:-/usr/bin/gcc} CXX=${CXX:-/usr/bin/g++} \
        LDFLAGS='-Wl,-rpath,$CONDA_PREFIX_PATH/lib -L$CONDA_PREFIX_PATH/lib' \
        ./configure --with-python3='$PYTHON_BIN' --without-perl --disable-doc --prefix='$CONDA_PREFIX_PATH' >/dev/null 2>&1 && \
        CC=${CC:-/usr/bin/gcc} CXX=${CXX:-/usr/bin/g++} \
        make -s -j${JOBS} >/dev/null 2>&1 && \
        make install >/dev/null 2>&1 \
    "

    rm -rf "$TMPDIR_VR"
fi

if [ ! -d "deepmirgene_src" ]; then
    git clone https://github.com/eleventh83/deepMiRGene.git deepmirgene_src
fi
