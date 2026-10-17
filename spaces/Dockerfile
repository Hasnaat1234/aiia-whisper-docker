# syntax=docker/dockerfile:1.6

# Hugging Face Spaces — single-container variant.
#
# HF Spaces only allows ONE container per Space, so this image merges the
# inference (Whisper) and the Gradio UI into one process. The local Docker
# Compose setup (../docker-compose.yml) keeps them in separate containers;
# this Dockerfile is only for the HF Spaces deployment.

FROM python:3.12-slim

ARG TORCH_VERSION=2.4.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG MODEL_ID=openai/whisper-tiny

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    MODEL_ID=${MODEL_ID}

WORKDIR /app

# ffmpeg for librosa audio decoding.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
 && pip install -r requirements.txt

# HF Spaces runs containers as a non-root user by default (uid=1000).
RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /opt/hf \
 && chown -R appuser:appuser /opt/hf
USER appuser

# Pre-download model into the image so first request is fast.
RUN python -c "import os; \
from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
mid = os.environ['MODEL_ID']; \
WhisperProcessor.from_pretrained(mid); \
WhisperForConditionalGeneration.from_pretrained(mid)"

COPY app.py ./

EXPOSE 7860

CMD ["python", "app.py"]
