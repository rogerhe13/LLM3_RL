# RLHF Assignment Dockerfile
# PyTorch with CUDA support for training language models

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV HF_HOME=/workspace/.cache/huggingface

# Create cache directory
RUN mkdir -p /workspace/.cache/huggingface

# Copy requirements first for better caching
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . /workspace/

# Make shell scripts executable
RUN chmod +x /workspace/part1/run_part1.sh \
    /workspace/part2/run_part2.sh \
    /workspace/part3/run_part3.sh \
    /workspace/part4/run_part4.sh 2>/dev/null || true

# Set default working directory for running experiments
WORKDIR /workspace

# Default command - start bash shell
CMD ["/bin/bash"]
