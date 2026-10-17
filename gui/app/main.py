"""Gradio web UI for the Whisper speech-to-text service.

This container holds NO machine-learning dependencies. It just renders a
small web form, accepts an audio file from the user, and POSTs it to the
inference container's `/transcribe` endpoint over the `dl-net` bridge network.
"""

from __future__ import annotations

import logging
import os

import gradio as gr
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("whisper-gui")

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://inference:8000")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "180"))


def transcribe(audio_file: str | None) -> str:
    """POST the audio file to the inference service and return the transcript.

    `audio_file` is a local path produced by Gradio's audio component.
    """
    if not audio_file:
        return "Please record or upload an audio clip first."

    log.info("Forwarding %s to %s/transcribe", audio_file, INFERENCE_URL)
    try:
        with open(audio_file, "rb") as f:
            files = {
                "file": (
                    os.path.basename(audio_file),
                    f,
                    "audio/wav",
                )
            }
            r = httpx.post(
                f"{INFERENCE_URL}/transcribe",
                files=files,
                timeout=REQUEST_TIMEOUT,
            )
        r.raise_for_status()
        data = r.json()
        return data.get("text", "")
    except httpx.HTTPStatusError as exc:
        log.error("Inference returned %s: %s", exc.response.status_code, exc.response.text)
        return f"Inference error ({exc.response.status_code}): {exc.response.text}"
    except httpx.RequestError as exc:
        log.error("Request to inference failed: %s", exc)
        return f"Could not reach inference service at {INFERENCE_URL}: {exc}"


# --------------------------------------------------------------------------- #
# Gradio interface
# --------------------------------------------------------------------------- #
demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(
        type="filepath",
        label="Record or upload audio (wav / mp3 / m4a / flac)",
    ),
    outputs=gr.Textbox(
        label="Transcription",
        lines=6,
        placeholder="The transcribed text will appear here …",
    ),
    title="Whisper Speech-to-Text",
    description=(
        "Record audio from your microphone or upload a file. The GUI container "
        "forwards it to the FastAPI inference container running "
        "`openai/whisper-tiny` and displays the returned transcription."
    ),
    article=(
        f"Internal endpoint: `{INFERENCE_URL}/transcribe`  \n"
        "This UI lives in a separate container with no ML dependencies."
    ),
    allow_flagging="never",
    examples=None,
)

if __name__ == "__main__":
    # server_name=0.0.0.0 is required so the port can be published to the host.
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        prevent_thread_lock=False,
    )
