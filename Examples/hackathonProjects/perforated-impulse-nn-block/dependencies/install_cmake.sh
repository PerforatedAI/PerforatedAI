#!/bin/bash
set -e

UNAME=`uname -m`

mkdir -p /opt/cmake
cd /opt/cmake
if [ "$UNAME" == "aarch64" ]; then
    wget https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3-linux-aarch64.sh
    sh cmake-3.21.3-linux-aarch64.sh --prefix=/opt/cmake --skip-license
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
    rm /opt/cmake/cmake-3.21.3-linux-aarch64.sh
else
    wget https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3-linux-x86_64.sh
    sh cmake-3.21.3-linux-x86_64.sh --prefix=/opt/cmake --skip-license
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
    rm /opt/cmake/cmake-3.21.3-linux-x86_64.sh
fi
