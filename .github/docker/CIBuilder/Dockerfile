FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"

# Update the package list and install prerequisites
RUN apt-get update && \
    apt-get install -y software-properties-common git curl build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget llvm libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
        libgdbm-dev liblz4-dev libsnappy-dev gcc-aarch64-linux-gnu && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash

# Initialize pyenv and install Python
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /root/.bashrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.bashrc && \
    echo 'eval "$(pyenv init --path)"' >> /root/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> /root/.bashrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> /root/.bashrc

# Install a specific Python version
RUN /root/.pyenv/bin/pyenv install 3.6.15 && \
    /root/.pyenv/bin/pyenv global 3.6.15

# Install pip for the installed Python version
RUN /root/.pyenv/shims/python -m ensurepip

# Install patchelf (the Python package)
#RUN /root/.pyenv/shims/pip install patchelf

# Set the default command to run when the container starts
CMD ["/bin/bash"]
