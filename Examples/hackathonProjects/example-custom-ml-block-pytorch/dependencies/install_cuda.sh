 #!/bin/bash
set -ex

UNAME=`uname -m`

if [ "$UNAME" == "aarch64" ]; then
    echo "Skipping CUDA on AARCH64..."
else
    # Update the CUDA Linux GPG Repository Key
    # See: https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
    apt-key del 7fa2af80
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        libcublas-${CUDA/./-} \
        cuda-nvrtc-${CUDA/./-} \
        libcufft-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcusparse-${CUDA/./-} \
        curl \
        libcudnn${CUDNN_MAJOR_VERSION}-cuda-12 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

    [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION} \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }
fi