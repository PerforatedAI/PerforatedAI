# Use PyTorch base image with CUDA pre-installed for faster builds
FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS base

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
SHELL ["/bin/bash", "-c"]

# Install system dependencies (things that rarely change)
RUN apt-get update && apt-get install -y \
    git \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy and install Python dependencies (changes occasionally)
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Copy application files last (changes frequently)
COPY train.py ./
COPY parameters.json ./

ENTRYPOINT ["python3", "-u", "train.py"]

