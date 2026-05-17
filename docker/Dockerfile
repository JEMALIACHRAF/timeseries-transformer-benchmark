# docker/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    numpy pandas scipy \
    fastapi uvicorn pydantic \
    mlflow \
    xgboost \
    shap \
    torch --extra-index-url https://download.pytorch.org/whl/cpu \
    neuralforecast \
    azure-storage-file-datalake azure-identity \
    && pip install --no-cache-dir \
    great-expectations plotly

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
