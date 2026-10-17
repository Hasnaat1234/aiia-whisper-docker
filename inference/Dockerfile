# syntax=docker/dockerfile:1.6

# --- Base image ---------------------------------------------------------------
# slim variant keeps the image small; python:3.12 is the current stable line.
FROM python:3.12-slim

# --- Build-time arguments -----------------------------------------------------
# These are baked into the image at build time. Override via docker-compose.yml
# build.args if you want a different torch wheel or model.
ARG TORCH_VERSION=2.4.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG MODEL_ID=openai/whisper-tiny

# --- Environment --------------------------------------------------------------
# DEBIAN_FRONTEND=noninteractive + PIP_NO_CACHE_DIR keep the build non-blocking
# and small. PYTHONUNBUFFERED ensures logs appear in real time on `docker logs`.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    MODEL_ID=${MODEL_ID}

WORKDIR /app

# --- Layer 1: OS packages (rarely changes, highly cacheable) -----------------
# ffmpeg is required by librosa to decode mp3/m4a/flac. curl is for HEALTHCHECK.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

# --- Layer 2: Python deps (changes when requirements.txt changes) --------------
# Install torch first from the (CPU or CUDA) index, then everything else.
# Putting torch in its own RUN means a requirement bump on `librosa` does NOT
# re-download the ~200 MB torch wheel.
COPY requirements.txt ./
RUN pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
 && pip install -r requirements.txt

# --- Layer 3: Non-root user (security best practice) -------------------------
RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /opt/hf /models \
 && chown -R appuser:appuser /opt/hf /models
USER appuser

# --- Layer 4: Pre-download the model into the image --------------------------
# Doing this BEFORE copying app/ means editing main.py does NOT invalidate
# this expensive layer. Whisper-tiny is ~150 MB on first download.
RUN python -c "import os; \
from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
mid = os.environ['MODEL_ID']; \
WhisperProcessor.from_pretrained(mid); \
WhisperForConditionalGeneration.from_pretrained(mid)"

# --- Layer 5: Application code (changes most often, last) ---------------------
COPY train.py ./
COPY app ./app

EXPOSE 8000

# --- Healthcheck: lets the GUI container wait for inference to be ready -------
# start-period=90s gives the model time to load on slow machines.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# --- Default command ----------------------------------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
