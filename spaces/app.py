"""Whisper Speech-to-Text — Hugging Face Spaces single-container variant.

Merges the inference (Whisper) and the Gradio UI into a single process
because HF Spaces allows only one container per Space.

The local Docker Compose setup (../docker-compose.yml) keeps them in
separate containers; this file is only used for the HF Spaces deployment.
"""

from __future__ import annotations

import logging
import os

import gradio as gr
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("whisper-spaces")

MODEL_ID = os.environ.get("MODEL_ID", "openai/whisper-tiny")
SAMPLE_RATE = 16_000

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Loading %s on %s ...", MODEL_ID, device)
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
model.eval()
log.info("Model ready.")


def transcribe(audio_file: str | None) -> str:
    if not audio_file:
        return "Please upload or record an audio clip first."
    try:
        audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:  # noqa: BLE001
        return f"Failed to decode audio: {exc}"
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
    with torch.inference_mode():
        token_ids = model.generate(inputs, max_new_tokens=400)
    return processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Record or upload audio"),
    outputs=gr.Textbox(label="Transcription", lines=6),
    title="Whisper Speech-to-Text",
    description=(
        "A small Whisper-tiny speech-to-text demo deployed on Hugging Face Spaces. "
        "Record audio from your microphone or upload a file; the model returns a "
        "transcription."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
