FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and configs
COPY service/ ./service/
COPY configs/ ./configs/
COPY config.yaml .
COPY params.yaml .
COPY dvc.lock .

# SQLite database and data directory mount points
VOLUME /app/data
VOLUME /app/mlflow.db

EXPOSE 8000

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
