FROM nvcr.io/nvidia/pytorch:26.02-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 \
    libgl1 libglib2.0-0 \
    && apt-get update \
    && rm -rf /var/lib/apt/lists/*

RUN  apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    ca-certificates \
    build-essential \
    git \
    vim \
    tmux \
    unzip \
    zip \
    tar \
    nano \
    less \
    htop \
    net-tools \
    iputils-ping \
    openssh-client \
    rsync \
    screen \
    sudo \
    locales \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
 && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
 && ln -sf /usr/bin/pip3 /usr/bin/pip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel uv
#  && /opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/nightly/cu129

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8888

CMD ["/bin/bash"]

WORKDIR /workspace