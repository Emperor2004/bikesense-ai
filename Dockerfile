# syntax=docker/dockerfile:1
# ═══════════════════════════════════════════════════════════════════════════════
# BikeSense AI — Production Dockerfile for Amazon EC2 Deployment
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

LABEL maintainer="BikeSense AI Team"
LABEL description="Citi Bike hourly departure demand forecasting microservice"

# Environment settings for python & container execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

WORKDIR /app

# Install minimal OS packages & dependencies (curl for EC2 healthchecks, git for lineage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code, pipeline configs, and metadata
COPY app/ ./app/
COPY service/ ./service/
COPY configs/ ./configs/
COPY config.yaml .
COPY params.yaml .
COPY dvc.lock .

# Bundle the model registry database and model artifacts inside the Docker image
# This ensures self-contained inference on Amazon EC2 without external runtime dependencies
COPY mlflow.db /app/mlflow.db
COPY mlruns/ /app/mlruns/

# Copy processed reference features for station inference fallback & metadata
COPY data/processed/ /app/data/processed/

# Create runtime prediction directory with appropriate permissions
RUN mkdir -p /app/data/predictions && chmod -R 777 /app/data

# Expose HTTP port for EC2 inbound security group
EXPOSE 8000

# Docker healthcheck compatible with EC2 instance & ALB target group health monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production ASGI server launch with multiple workers for EC2 concurrency
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
