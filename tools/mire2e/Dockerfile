FROM python:3.9-slim

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

WORKDIR /opt/mire2e

RUN pip install --no-cache-dir torch>=1.7 biopython>=1.78 scikit-learn>=0.23 tqdm miRe2e

RUN python -c "from miRe2e import MiRe2e; MiRe2e(pretrained='hsa'); MiRe2e(pretrained='animals'); print('Models downloaded')"

COPY inference.py ./

WORKDIR /work

ENTRYPOINT ["python", "/opt/mire2e/inference.py"]
