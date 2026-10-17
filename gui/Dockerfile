# syntax=docker/dockerfile:1.6

# --- Base image ---------------------------------------------------------------
FROM python:3.12-slim

# --- Environment --------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- Layer 1: OS packages (only curl for HEALTHCHECK) -------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# --- Layer 2: Python deps ----------------------------------------------------
COPY requirements.txt ./
RUN pip install -r requirements.txt

# --- Layer 3: Non-root user --------------------------------------------------
RUN useradd --create-home --uid 1000 appuser
USER appuser

# --- Layer 4: Application code (last — changes most often) -------------------
COPY app ./app

EXPOSE 7860

# --- Healthcheck --------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:7860/ || exit 1

# --- Default command ----------------------------------------------------------
CMD ["python", "app/main.py"]
