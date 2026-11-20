
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directory for runtime artifacts
RUN mkdir -p data/artifacts

# MLflow Model Registry connection
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000
ENV MLFLOW_MODEL_NAME=chest-cancer-classifier
ENV MLFLOW_MODEL_STAGE=Production

# API configuration
EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
