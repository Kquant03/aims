FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install CUDA-specific packages
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p data/states data/backups data/memories logs

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose ports
EXPOSE 8000 8765

# Run the application
CMD ["python3", "-m", "src.main"]