# ============================================================
# Dubai Retail Intelligence Agent — API Container
# ============================================================
# Base: Python 3.12 slim (smaller image, no unnecessary OS packages)
# 
# ENGINEERING DECISION: python:3.12-slim over python:3.12-alpine
# Alpine uses musl libc — many ML packages (torch, numpy) don't have
# Alpine wheels and require compiling from source (~20 min build).
# slim uses glibc — wheels install in seconds.
# Trade-off: slim is ~50MB larger than alpine. Acceptable for an AI service.

FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# ── System dependencies ──────────────────────────────────────────────────
# libgomp1: required by sentence-transformers (OpenMP for parallel embedding)
# build-essential: needed for some pip packages that compile C extensions
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────
# Copy requirements first — Docker caches this layer.
# If you only change code (not requirements), Docker skips the pip install
# step on rebuild. This makes rebuilds fast (~5 seconds vs ~5 minutes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────
COPY . .

# ── Create directories that need to exist at runtime ─────────────────────
# These will be overridden by Docker volumes, but need to exist in the image
RUN mkdir -p .model_cache .chroma_db data/synthetic

# ── Port ──────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────
# Docker will mark the container unhealthy if /health returns non-200.
# This is what allows docker compose to know when the API is truly ready.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Startup command ───────────────────────────────────────────────────────
# --host 0.0.0.0: listen on all interfaces (required inside Docker)
# --port 8000: matches EXPOSE above
# --workers 1: single worker for Phase 1 (the embedding model is loaded
#              once per worker — multiple workers = multiple model copies in RAM)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
