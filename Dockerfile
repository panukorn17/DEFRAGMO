FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "source activate DEFRAGMO" > ~/.bashrc
ENV PATH /opt/conda/envs/DEFRAGMO/bin:$PATH

RUN conda install -n DEFRAGMO \
    "pytorch::pytorch=2.2.0=py3.11_cuda11.8_cudnn8.7.0_0" \
    "pytorch::torchvision=0.17.0=py311_cu118" \
    "pytorch::torchaudio=2.2.0=py311_cu118" \
    "pytorch::pytorch-cuda=11.8" \
    --no-deps

COPY src/ ./src/
COPY data/ ./data/

