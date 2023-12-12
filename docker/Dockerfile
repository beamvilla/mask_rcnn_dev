FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04
RUN nvidia-smi

# SYSTEM
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils cmake zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget libbz2-dev \
    wget curl vim git ca-certificates kmod \
 && rm -rf /var/lib/apt/lists/*
 

RUN add-apt-repository ppa:deadsnakes/ppa && apt install python3.7 && rm -rf /var/lib/apt/lists/*
RUN python3.7 --version

WORKDIR /root
ADD /docker/requirements.txt /root/requirements.txt
RUN pip3 install --user --no-cache-dir -r requirements.txt

RUN mkdir -p /mask_rcnn_dev
WORKDIR /mask_rcnn_dev