"""FastAPI inference server for Whisper speech-to-text.

Endpoints
---------
GET  /health       -> liveness + model/device info
POST /transcribe   -> multipart audio file -> {"text": "...", "model": ..., "duration_s": ...}
GET  /             -> small landing page

Environment
-----------
MODEL_ID    Hugging Face model id (default "openai/whisper-tiny")
MODEL_PATH  Optional path to a fine-tuned model on disk; takes precedence over MODEL_ID
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict

import librosa
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import transformers

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("whisper-inference")

MODEL_ID = os.environ.get("MODEL_ID", "openai/whisper-tiny")
MODEL_PATH = os.environ.get("MODEL_PATH")  # if set, load from local path
SAMPLE_RATE = 16_000


def _resolve_source() -> str:
    """Return the local path if set and exists, else the HF model id."""
    if MODEL_PATH and os.path.isdir(MODEL_PATH):
        return MODEL_PATH
    return MODEL_ID


# --------------------------------------------------------------------------- #
# App state (populated on startup)
# --------------------------------------------------------------------------- #
_state: Dict[str, Any] = {"processor": None, "model": None, "device": None}

app = FastAPI(
    title="Whisper Speech-to-Text Inference",
    version="1.0.0",
    description="Run Whisper transcription inside a Docker container.",
)


@app.on_event("startup")
async def _load_model() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src = _resolve_source()
    log.info("Loading Whisper from %s on %s ...", src, device)
    _state["processor"] = WhisperProcessor.from_pretrained(src)
    _state["model"] = WhisperForConditionalGeneration.from_pretrained(src).to(device)
    _state["model"].eval()
    _state["device"] = device
    log.info("Whisper loaded. device=%s torch=%s transformers=%s",
             device, torch.__version__, transformers.__version__)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return (
        "<h3>Whisper Speech-to-Text Inference Service</h3>"
        "<ul>"
        "<li><a href='/docs'>Swagger UI</a></li>"
        "<li><a href='/health'>Health check</a></li>"
        "<li><code>POST /transcribe</code> with a multipart <code>file</code> field</li>"
        "</ul>"
    )


@app.get("/health")
async def health() -> JSONResponse:
    if _state["model"] is None:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "model": _resolve_source()},
        )
    return {
        "status": "ok",
        "model": _resolve_source(),
        "device": _state["device"],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> JSONResponse:
    if _state["model"] is None:
        raise HTTPException(503, "Model still loading; retry shortly.")

    # Accept any audio/* MIME, but be lenient with octet-stream / unknown types.
    ctype = (file.content_type or "").lower()
    if ctype and not ctype.startswith("audio/") and ctype != "application/octet-stream":
        log.warning("Unexpected content-type: %s — will still try to decode.", ctype)

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")

    # Decode audio to a 16 kHz mono float32 numpy array using librosa + ffmpeg.
    try:
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:  # noqa: BLE001 — surfaced as a 400 to the caller
        raise HTTPException(400, f"Failed to decode audio: {exc}") from exc

    # Whisper feature extraction + forward pass.
    proc = _state["processor"]
    inputs = proc(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
    inputs = inputs.to(_state["device"])
    with torch.inference_mode():
        token_ids = _state["model"].generate(inputs, max_new_tokens=400)
    text = proc.batch_decode(token_ids, skip_special_tokens=True)[0].strip()

    return {
        "text": text,
        "model": _resolve_source(),
        "duration_s": round(len(audio) / SAMPLE_RATE, 3),
        "device": _state["device"],
    }
