# Stage 1: Builder
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv pip install --system --no-cache \
    fastapi>=0.115.0 \
    "uvicorn[standard]>=0.34.0" \
    httpx>=0.27.0 \
    numpy>=1.24.0 \
    onnxruntime>=1.17.0 \
    "psycopg[binary]>=3.1.0" \
    sqlalchemy>=2.0.0

# Stage 2: Runtime
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

COPY --chown=appuser:appuser api/ api/
COPY --chown=appuser:appuser results/lightgbm_optimized.onnx results/lightgbm_optimized.onnx

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
