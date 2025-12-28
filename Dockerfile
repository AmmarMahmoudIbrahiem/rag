FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# System dependencies (FAISS + build)
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (GOOD PRACTICE)
COPY src/requirements.txt .

# Install dependencies
RUN pip install \
    --default-timeout=100 \
    --retries 10 \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy application code
COPY src/ ./src/

# Data dir for FAISS
RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/welcome || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
