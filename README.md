# AIIA — Deployment of Deep Learning Architectures in Docker Containers

A Docker Compose project that deploys a **Whisper speech-to-text** model split across
two logically isolated containers:

| Container   | Role                                                          | Port  | Image size (approx.) |
|-------------|---------------------------------------------------------------|-------|----------------------|
| `inference` | FastAPI server running `openai/whisper-tiny` transcription    | 8000  | ~600 MB              |
| `gui`       | Gradio web UI — no ML deps, forwards HTTP to `inference`       | 7860  | ~250 MB              |

The two containers communicate over an isolated Docker bridge network (`dl-net`).
The GUI container contains **no** machine-learning dependencies — it only posts the
uploaded audio file to the inference container's `/transcribe` endpoint and displays
the returned text. This satisfies the assignment requirement that the graphical
interface and the inference logic live in separate containers.

An optional `train.py` script demonstrates a few-epoch fine-tuning run inside the
inference container, and a separate `spaces/` directory packages the whole app as a
single-container Docker image suitable for upload to Hugging Face Spaces.

---

## 1. Prerequisites

| Tool                       | Why                                            | Install link                                           |
|----------------------------|------------------------------------------------|--------------------------------------------------------|
| Docker Desktop ≥ 4.30      | Runs the containers on Windows/macOS/Linux      | <https://www.docker.com/products/docker-desktop>      |
| WSL2 (Windows only)        | Docker Desktop backend on Windows              | <https://learn.microsoft.com/windows/wsl/install>     |
| `nvidia-container-toolkit`| Only if you want the GPU bonus (optional)       | <https://docs.nvidia.com/datacenter/cloud-native/>    |

Verify the install (in PowerShell or WSL bash):

```bash
docker --version
docker compose version
docker info
```

On Windows, make sure **Use the WSL 2 based engine** is enabled in
Docker Desktop → Settings → General. Assign at least **4 GB RAM** and
**2 CPUs** to the WSL2 backend (Settings → Resources → WSL Integration).

---

## 2. Repository layout

```
aiia-whisper-docker/
├── README.md                    # this file
├── docker-compose.yml           # main compose file (CPU)
├── docker-compose.gpu.yml       # optional GPU override (NVIDIA Container Toolkit)
├── .gitignore
├── .dockerignore
├── inference/                   # container 1: FastAPI + Whisper
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── train.py                 # optional few-epoch fine-tune
│   └── app/
│       └── main.py              # FastAPI server (whisper inference)
├── gui/                         # container 2: Gradio UI (no ML deps)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       └── main.py              # Gradio interface → POSTs to inference
└── spaces/                      # Hugging Face Spaces single-container variant
    ├── Dockerfile
    ├── app.py
    ├── requirements.txt
    └── README.md
```

---

## 3. Build and run

From the repository root:

```bash
# Build both images
docker compose build

# Start in detached mode
docker compose up -d

# Wait for inference to become "healthy" (model load takes ~30-60 s)
docker compose ps
```

When `aiia-whisper-inference` shows `(healthy)`, open:

- **Gradio GUI** → <http://localhost:7860>
- **FastAPI docs (Swagger)** → <http://localhost:8000/docs>
- **Health check** → <http://localhost:8000/health>

Stop and remove everything (the named volume `models` is kept):

```bash
docker compose down          # stop + remove containers
docker compose down -v       # also delete the models volume
```

---

## 4. Test inference from the terminal

Record a short wav file (any tool works — Windows Sound Recorder, Audacity, or
`ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 5 sample.wav`) and run:

```bash
# Inference directly via FastAPI
curl -s -X POST http://localhost:8000/transcribe `
  -F "file=@sample.wav" | python -m json.tool
```

If `curl` is unavailable on Windows, use `Invoke-RestMethod` in PowerShell:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/transcribe `
  -Method Post -Form @{ file = Get-Item .\sample.wav }
```

Or just open the Gradio UI at <http://localhost:7860> and click **Record**.

---

## 5. Optional: fine-tune Whisper inside the inference container

`train.py` runs a tiny synthetic few-epoch update to demonstrate the training
pipeline works end-to-end inside the container (no internet access needed):

```bash
# Run 1 epoch on 32 synthetic samples (very fast, demonstrates the pipeline)
docker compose exec inference python train.py --epochs 1 --train-samples 32

# The fine-tuned model is saved to the `models` volume at /models/finetuned-whisper.
# To serve it, restart the inference container with MODEL_PATH set:
$env:MODEL_PATH="/models/finetuned-whisper"
docker compose up -d --force-recreate inference
```

> Note: the synthetic dataset is just for pipeline demonstration. For a real
> fine-tune, replace `train.py`'s `TinyDataset` with a real Common Voice / LibriSpeech
> loader.

---

## 6. Optional: run on GPU (NVIDIA Container Toolkit)

If you have an NVIDIA GPU and `nvidia-container-toolkit` installed, merge the
GPU override file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

The override swaps the CPU-only PyTorch wheel for the CUDA build
(`https://download.pytorch.org/whl/cu121`) and reserves all NVIDIA GPUs for the
inference container. Whisper automatically runs on CUDA; check via:

```bash
docker compose exec inference curl -s http://localhost:8000/health
# {"device": "cuda", ...}
```

---

## 7. Optional: deploy to Hugging Face Spaces

The `spaces/` directory is a self-contained variant that runs the inference and
the Gradio UI inside a single container (because HF Spaces only allows one
container per Space). To deploy:

1. Create a new Space at <https://huggingface.co/new-space> (SDK: **Docker**).
2. Clone the Space locally:

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<your-space>
   ```
3. Copy `spaces/Dockerfile`, `spaces/app.py`, `spaces/requirements.txt`, and
   `spaces/README.md` into the Space repo (at the root, replacing any default files).
4. Push:

   ```bash
   git add .
   git commit -m "Initial Whisper STT deployment"
   git push
   ```

HF Spaces builds the Dockerfile automatically. Once built, the Space exposes
the Gradio UI at `https://<your-username>-<your-space>.hf.space`.

---

## 8. Dockerfile design notes (for the report)

The inference `Dockerfile` follows several best practices explicitly listed in
the assignment:

- **Cache ordering** — least-frequently changed layers come first (`apt-get`,
  `pip install torch`, model pre-download), most-frequently changed layers last
  (`COPY app ./app`). Editing `app/main.py` does NOT invalidate the layer that
  downloads the 150 MB Whisper weights.
- **`DEBIAN_FRONTEND=noninteractive`** plus `-y` flags on every `apt-get install`
  so the build never blocks waiting for user input.
- **`--no-install-recommends`** + `rm -rf /var/lib/apt/lists/*` keeps the apt layer
  small.
- **Non-root user** (`appuser`, uid 1000) — the FastAPI server does not run as root.
- **Pre-baked model** — the Whisper weights are downloaded at build time so the
  container can run inference without internet access at runtime.
- **`HEALTHCHECK`** on the inference container lets the GUI container wait for the
  model to finish loading via `depends_on: condition: service_healthy`.
- **`PIP_NO_CACHE_DIR=1`** avoids keeping installer caches in the image.

The `docker-compose.yml` enforces:

- **Resource limits** — inference capped at 4 GB / 2 CPUs, GUI at 1 GB / 1 CPU.
- **`shm_size`** — 512 MB for inference (PyTorch DataLoader-friendly), 256 MB for GUI.
- **`depends_on` with health condition** — the GUI will not start until inference
  reports healthy.
- **Bridge network `dl-net`** — isolates the two services from the host network.
- **Named volume `models`** — persists fine-tuned weights across container restarts.
- **`restart: unless-stopped`** — survives host reboots.

---

## 9. Troubleshooting

| Symptom                                          | Fix                                                                                          |
|--------------------------------------------------|----------------------------------------------------------------------------------------------|
| `inference` stays `unhealthy` for >90 s          | Model is still downloading on first build; bump `start_period` in the Dockerfile.            |
| GUI shows "Request failed: connection refused"   | `inference` is not healthy yet. Check `docker compose logs inference`.                       |
| `librosa.load` errors on upload                  | Make sure the file is real audio (wav/mp3/m4a/flac). `ffmpeg` is installed in the image.    |
| Out-of-memory on Windows                         | Increase Docker Desktop's WSL2 memory limit (Settings → Resources).                          |
| `nvidia-container-toolkit` not detected on GPU   | Make sure Docker Desktop → Settings → Resources → NVIDIA GPU is enabled.                    |

---

## 10. Cleaning up

```bash
docker compose down -v          # remove containers + volume
docker rmi aiia/whisper-inference aiia/whisper-gui
docker system prune -f          # also drop dangling images/containers (optional)
```

---

## License

This project is an academic submission for the AIIA course. The pre-trained
Whisper model is © OpenAI, released under the MIT license via Hugging Face.
