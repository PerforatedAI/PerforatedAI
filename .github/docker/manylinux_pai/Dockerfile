FROM quay.io/pypa/manylinux2014_x86_64

# Install required libraries
RUN yum install -y glibc glibc-devel libstdc++ libstdc++-devel

# Install Node.js (modify the version as needed)
RUN curl -sL https://rpm.nodesource.com/setup_14.x | bash -
RUN yum install -y nodejs

# Your other commands go here...