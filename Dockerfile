# Adobe Round 1A: PDF Hierarchy Detector Docker Container
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create necessary directories
RUN mkdir -p input output models training_data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Make sure the main script is executable
RUN chmod +x main.py

# Default command - runs the complete pipeline
CMD ["python", "main.py"]

# For Adobe Round 1A evaluation
VOLUME ["/app/input", "/app/output"]
