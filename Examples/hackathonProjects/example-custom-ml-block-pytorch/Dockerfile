# syntax = docker/dockerfile:experimental@sha256:3c244c0c6fc9d6aa3ddb73af4264b3a23597523ac553294218c13735a2c6cf79
ARG UBUNTU_VERSION=22.04
ARG CUDA=12.9

FROM --platform=linux/amd64 nvidia/cuda:${CUDA}.0-base-ubuntu${UBUNTU_VERSION} AS base

ARG CUDA
ARG CUDNN=9.10.2.21-1
ARG CUDNN_MAJOR_VERSION=9
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=10.0.0-1
ARG LIBNVINFER_MAJOR_VERSION=10
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
SHELL ["/bin/bash", "-c"]

COPY dependencies/install_cuda.sh ./install_cuda.sh
RUN /bin/bash ./install_cuda.sh && rm install_cuda.sh

RUN apt update && apt install -y curl zip git lsb-release software-properties-common \
    apt-transport-https vim wget python3 python3-pip
RUN python3 -m pip install --upgrade pip==25.3

COPY dependencies/install_cmake.sh install_cmake.sh
RUN /bin/bash install_cmake.sh && rm install_cmake.sh

RUN apt update && apt install -y protobuf-compiler

RUN python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt ./

RUN pip3 install -r requirements.txt

RUN pip3 install tf_keras

COPY . ./
ENTRYPOINT ["python3", "-u", "train.py"]

