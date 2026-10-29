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

# Install minimal OS packages (curl for healthchecks, git for lineage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight Python serving requirements
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-serve.txt

# Copy application code, pipeline configs, and metadata
COPY app/ ./app/
COPY service/ ./service/
COPY configs/ ./configs/
COPY config.yaml .
COPY params.yaml .
COPY dvc.lock .

# Bundle the model registry database — required for serving
# (mlruns/ and mlflow.db must exist before building; they are populated locally)
COPY mlflow.db /app/mlflow.db
COPY mlruns/ /app/mlruns/

# Normalize host file URIs in mlflow.db to container paths
# Handles both file:D:/... (no triple slash) and file:///D:/... (triple slash) variants
RUN python -c "\
import sqlite3; \
conn = sqlite3.connect('/app/mlflow.db'); \
cur = conn.cursor(); \
cur.execute(\"UPDATE experiments SET artifact_location = REPLACE(REPLACE(artifact_location, 'file:D:/Projects/bikesense-ai/mlruns', 'file:///app/mlruns'), 'file:///D:/Projects/bikesense-ai/mlruns', 'file:///app/mlruns')\"); \
cur.execute(\"UPDATE runs SET artifact_uri = REPLACE(REPLACE(artifact_uri, 'file:D:/Projects/bikesense-ai/mlruns', 'file:///app/mlruns'), 'file:///D:/Projects/bikesense-ai/mlruns', 'file:///app/mlruns')\"); \
conn.commit(); \
conn.close()"

# Copy processed reference features for station inference fallback & metadata
COPY data/processed/ /app/data/processed/

# Create runtime prediction directory with appropriate permissions
RUN mkdir -p /app/data/predictions && chmod -R 777 /app/data

# Expose HTTP port for EC2 inbound security group
EXPOSE 8000

# Docker healthcheck compatible with EC2 instance & ALB target group health monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Production ASGI server launch
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
