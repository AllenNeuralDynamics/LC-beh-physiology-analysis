# hash:sha256:8d5d3da8954239b6d5b596e398a691544b9471cecd013aadb57a59f47ce13dd2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:v5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ENV CLAUDE_CODE_USE_BEDROCK=1
ENV AWS_REGION=us-west-1

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-ask-pass /

RUN conda install -y \
        ipykernel==6.30.1 \
    && conda clean -ya

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        awscli=1.22.34-1 \
        build-essential=12.9ubuntu3 \
        libgl1-mesa-glx=23.0.4-0ubuntu1~22.04.1 \
        libssl-dev=3.0.2-0ubuntu1.29 \
        pandoc=2.9.2.1-3ubuntu2 \
        pkg-config=0.29.2-1ubuntu3 \
        poppler-utils=22.02.0-2ubuntu0.13 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir uv==0.7.15

RUN pip install -U --no-cache-dir \
    aind-ephys-utils==0.0.15 \
    hdmf-zarr==0.11.0 \
    ipympl==0.9.3 \
    nbconvert==7.16.6 \
    matplotlib==3.9.1 \
    numpy==2.1.3 \
    open-ephys-python-tools==0.1.10 \
    pynwb==3.1.3 \
    pypdf2==3.0.1 \
    spikeinterface[full]==0.103.0 \
    statsmodels==0.14.5 \
    wavpack-numcodecs==0.2.2 \
    natsort==8.4.0 \
    harp-python@git+https://github.com/jsiegle/harp-python@c73984a18905bb084dbb5b634776a3b5e47b511e \
    aind-dynamic-foraging-data-utils==0.1.4 \
    dandi==0.74.0 \
    pdf2image==1.17.0 \
    seaborn==0.13.2 \
    trimesh==4.10.1 \
    k3d==2.17.0 \
    aind-s3-cache==0.1.0 \
    codeocean==0.14.0 \
    pandas==2.3.2 \
    scikit-image==0.25.2 \
    aind_data_schema==2.6.0 \
    openpyxl==3.1.5 \
    pycircstat2==0.1.15 \
    scanpy==1.12.1 \
    myterial==1.2.1 \
    rich==14.3.1 \
    brainrender==2.1.17 \
    brainglobe-atlasapi==2.3.0



COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall

