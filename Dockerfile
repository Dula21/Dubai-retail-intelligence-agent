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
#
# SECURITY: "COPY . ." copies everything the build context contains, so .dockerignore keeps
# .env (API keys), .git and generated data out of the image. Data is mounted at runtime.

FROM python:3.12-slim

WORKDIR /app

# PYTHONPATH: nodes import "services.*" at top level, the API imports "backend.*".
# PYTHONUNBUFFERED: logs appear immediately in "docker compose logs".
ENV PYTHONPATH=/app/backend \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

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
# step on rebuild. This makes rebuilds fast.
COPY requirements.txt .
# CPU-only torch first so sentence-transformers does not pull the CUDA build (no GPU on target hosts).
# Pinned to the version the working image resolved (pip show torch -> 2.14.1+cpu), so rebuilds match.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────
COPY . .

# ── Create directories that need to exist at runtime ─────────────────────
# These will be overridden by Docker volumes, but need to exist in the image
RUN mkdir -p .model_cache .chroma_db data/synthetic

# ── Port ──────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────
# Docker marks the container unhealthy if /health returns non-200.
# start-period covers the embedding model load (cold start).
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Startup command ───────────────────────────────────────────────────────
# --host 0.0.0.0: listen on all interfaces (required inside Docker)
# --workers 1: the embedding model is loaded once per worker; more workers = more model copies in RAM
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]