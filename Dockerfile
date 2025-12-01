# Use the Jetson Nano / JetPack 4.6 compatible base image
FROM nvcr.io/nvidia/l4t-base:r32.6.1

ENV DEBIAN_FRONTEND=noninteractive

# Install basic packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Create a working directory
WORKDIR /workspace

CMD ["bash"]
