---
title: Whisper Speech-to-Text
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Whisper Speech-to-Text — Hugging Face Spaces

A minimal speech-to-text demo running OpenAI's `whisper-tiny` model inside a
Docker container deployed on Hugging Face Spaces.

## How it works

1. The container is built from [`Dockerfile`](./Dockerfile), which:
   - Installs `torch` (CPU wheel by default), `transformers`, `librosa`, `gradio`.
   - Pre-downloads the `openai/whisper-tiny` weights at build time so the first
     request is fast.
2. `app.py` loads the model into memory at startup and exposes a Gradio
   interface on port 7860 (the HF Spaces default).
3. Users record audio from their microphone or upload a file; the model returns
   the transcription.

## Local testing

```bash
docker build -t whisper-spaces .
docker run --rm -p 7860:7860 whisper-spaces
# open http://localhost:7860
```

## Companion repo

This Space is a single-container variant of a larger two-container Docker
Compose project where the inference (FastAPI + Whisper) and the GUI (Gradio)
run in separate containers. See the parent repository for details.
