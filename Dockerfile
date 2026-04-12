# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install CPU-only torch first to avoid downloading 2+ GB of CUDA packages,
# then install the rest of the backend dependencies.
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend .

# Create non-root user for security
RUN useradd -m -u 1000 moviebuzz && chown -R moviebuzz:moviebuzz /app
USER moviebuzz

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run application
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers
