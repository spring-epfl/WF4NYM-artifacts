# Dockerfile for WF4NYM Artifact
# Privacy Enhancing Technologies Symposium (PETs) 2026
# Paper: "Website Fingerprinting on Nym: Attacks and Defenses"

FROM python:3.9-slim

# Set metadata
LABEL maintainer="eric.jolles@epfl.ch"
LABEL description="Artifact for Website Fingerprinting on Nym: Attacks and Defenses (PETs 2026)"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    less \
    curl \
    build-essential \
    patch \
    sudo \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir jupyter notebook ipywidgets

# Copy the artifact code
COPY captures/ /workspace/captures/
COPY correlation/ /workspace/correlation/
COPY feature_importance/ /workspace/feature_importance/
COPY WF_attacks/ /workspace/WF_attacks/
COPY ARTIFACT-APPENDIX.md /workspace/
COPY README.md /workspace/
COPY .dockerignore /workspace/
COPY docker-compose.yml /workspace/
COPY download_data.sh /workspace/

# Create necessary directories
RUN mkdir -p /workspace/data && \
    mkdir -p /workspace/output && \
    mkdir -p /workspace/tmp

# Download and setup ExplainWF framework
RUN cd /workspace/WF_attacks && \
    git clone https://github.com/explainwf-popets2023/explainwf-popets2023.github.io.git && \
    cd explainwf-popets2023.github.io/ml/code && \
    bash download.bash && \
    bash patch.bash && \
    patch -p3 < ../../../explainwf_modifications.patch && \
    cp ../../../train_test.py ./

# Download test dataset from Zenodo
RUN cd /workspace/data && \
    echo "Downloading test dataset from Zenodo..." && \
    wget -O data_test.zip "https://zenodo.org/records/17867461/files/data_test.zip?download=1" && \
    echo "Extracting test dataset..." && \
    unzip -q data_test.zip && \
    rm -f data_test.zip && \
    echo "Test dataset download complete!"

# NOTE: Full datasets are NOT downloaded during build to save space and time.
# Users should either:
# 1. Download datasets manually to data/ directory before building
# 2. Mount data directory as a volume
# 3. Download datasets after container starts
#
# To download datasets inside the container, run:
# cd /workspace/data && bash /workspace/download_data.sh

# Expose Jupyter port
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes

# Allow root to run sudo without password (for notebook commands)
RUN echo "root ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Fix permissions on workspace (will be applied at runtime for mounted volumes)
RUN chmod -R 777 /workspace

# Switch to root user and ensure permissions
USER root

# Default command: Start Jupyter notebook
CMD ["sh", "-c", "chmod -R 777 /workspace 2>/dev/null || true && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
