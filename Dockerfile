# =============================================================================
# fin-risk-engine — Fraud scoring API (FastAPI + Gradio)
# =============================================================================
# Model artifacts and MLflow DB are NOT baked in.
# They are volume-mounted at runtime via docker-compose.
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# System deps:
#   libgomp1 — OpenMP required by CatBoost and XGBoost
#   curl     — used by the HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependency layer (cached unless pyproject.toml changes) ──────────────────
# Copy only the package manifest first so the heavy pip install is cached
# independently of source code changes.
COPY pyproject.toml README.md ./
# Create minimal src stub so setuptools can discover the package during install
RUN mkdir -p src && pip install --no-cache-dir ".[ml,serve,viz,tracking]"

# ── Application source ───────────────────────────────────────────────────────
# Copied after deps so source edits don't bust the dependency cache layer.
COPY src/ src/
COPY config/ config/

# ── Runtime directories (volume-mounted in production) ───────────────────────
# Directories exist in the image so the container starts even without mounts.
RUN mkdir -p outputs/models \
             outputs/artifacts \
             outputs/catboost_info \
             mlruns \
             data/processed

ENV PYTHONPATH=/app

EXPOSE 8000

# Health check — waits 60 s for model load before probing
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
