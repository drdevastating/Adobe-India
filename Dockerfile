# Adobe Round 1A: PDF Hierarchy Detection - Production Dockerfile
# Optimized for competition constraints: <10s processing, <200MB model, CPU-only

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm --quiet

# Create necessary directories
RUN mkdir -p input output models logs

# Copy application code
COPY *.py ./
COPY *.yaml ./
COPY *.yml ./

# Copy utility scripts
COPY setup.py ./

# Create a script to download and cache the ML model
RUN echo 'import torch\n\
from transformers import AutoTokenizer, AutoModelForSequenceClassification\n\
\n\
# Download DistilBERT model (lightweight, ~95MB)\n\
model_name = "distilbert-base-uncased"\n\
print("Downloading DistilBERT model...")\n\
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/app/models")\n\
model = AutoModelForSequenceClassification.from_pretrained(\n\
    model_name,\n\
    num_labels=4,\n\
    cache_dir="/app/models"\n\
)\n\
\n\
# Save locally to ensure consistent access\n\
model.save_pretrained("/app/models/distilbert")\n\
tokenizer.save_pretrained("/app/models/distilbert")\n\
print("Model cached successfully!")' > /app/download_model.py

# Run the model download script
RUN python /app/download_model.py && rm /app/download_model.py

# Set proper permissions
RUN chmod +x *.py

# Optimize for CPU execution (no GPU dependencies)
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Health check script
RUN echo '#!/bin/bash\npython -c "import torch; import transformers; import fitz; print(\"System ready\")"' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Expose no ports (offline processing)
# EXPOSE 8080

# Default volumes for input/output
VOLUME ["/app/input", "/app/output"]

# Main entry point
ENTRYPOINT ["python", "main.py"]

# Health check to ensure all dependencies are working
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Multi-stage build option for even smaller image (optional)
# FROM python:3.9-slim as production
# WORKDIR /app
# COPY --from=0 /app /app
# COPY --from=0 /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
# CMD ["python", "main.py"]

# Labels for Adobe competition
LABEL maintainer="Adobe Round 1A Submission"
LABEL version="2.0.0"
LABEL description="Improved PDF Hierarchy Detection with Quality Control"
LABEL constraints="<10s processing, <200MB model, CPU-only, offline"

# Build and run instructions:
# docker build --platform linux/amd64 -t adobe-round1a-improved .
# docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none adobe-round1a-improved