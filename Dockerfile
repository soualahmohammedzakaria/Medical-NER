# ---------- Medical-NER API ----------
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (some torch / tokenizers wheels need gcc)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy application source
COPY config/  config/
COPY src/     src/
COPY api/     api/
COPY setup.py .

# Copy model checkpoint (must exist before building)
COPY outputs/models/best/ outputs/models/best/

ENV PORT=8000
EXPOSE 8000

CMD python -m api.main --host 0.0.0.0 --port $PORT
