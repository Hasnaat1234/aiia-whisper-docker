"""Optional few-epoch fine-tune of Whisper-tiny inside the inference container.

This is a *demonstration* training pipeline: it uses a tiny SYNTHETIC dataset
(random audio + a constant transcription) so the script runs without internet
access and finishes in seconds. It proves that:

  * PyTorch + transformers training loop works inside the container,
  * The trained weights can be saved to the mounted `/models` volume,
  * The inference server can then load the fine-tuned weights via MODEL_PATH.

For a real fine-tune, replace `TinyDataset` with a HF `datasets` loader for
Common Voice or LibriSpeech and increase the epoch count.

Usage (inside the running inference container):

    docker compose exec inference python train.py --epochs 1 --train-samples 32

After training, restart the inference container pointing at the new weights:

    # PowerShell:
    $env:MODEL_PATH="/models/finetuned-whisper"
    docker compose up -d --force-recreate inference

    # bash:
    MODEL_PATH=/models/finetuned-whisper docker compose up -d --force-recreate inference
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("whisper-train")

MODEL_ID = os.environ.get("MODEL_ID", "openai/whisper-tiny")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/models/finetuned-whisper")
SAMPLE_RATE = 16_000


class TinyDataset(Dataset):
    """A minimal synthetic dataset for pipeline demonstration only.

    Each sample is 5 seconds of random noise paired with the constant label
    "hello world". After 1 epoch the model has no useful new knowledge — the
    point is to prove the training step itself runs end-to-end inside the
    container, and that the saved weights can be reloaded by the server.
    """

    def __init__(self, processor: WhisperProcessor, n: int = 32,
                 sample_rate: int = SAMPLE_RATE, seed: int = 0):
        self.processor = processor
        self.n = n
        self.sample_rate = sample_rate
        rng = np.random.default_rng(seed)
        # 5 seconds of random audio per sample
        self.audio = [
            rng.standard_normal(sample_rate * 5).astype("float32") for _ in range(n)
        ]
        self.label = "hello world"

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        feats = self.processor(
            self.audio[idx],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).input_features[0]
        labels = self.processor.tokenizer(self.label).input_ids
        return {"input_features": feats, "labels": torch.tensor(labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny Whisper fine-tune demo")
    parser.add_argument("--epochs", type=float, default=1.0,
                        help="Number of training epochs (float, default 1.0)")
    parser.add_argument("--train-samples", type=int, default=32,
                        help="Number of synthetic training samples (default 32)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device train batch size (default 4)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Where to save the fine-tuned model (default {OUTPUT_DIR})")
    args = parser.parse_args()

    log.info("Loading model %s ...", MODEL_ID)
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    # Don't force the language ids — let the model learn freely.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    dataset = TinyDataset(processor, n=args.train_samples)
    log.info("Dataset: %d samples", len(dataset))

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=1e-4,
        logging_steps=4,
        save_steps=10**9,           # don't save intermediates
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.feature_extractor,
    )

    log.info("Starting training ...")
    trainer.train()
    log.info("Saving model to %s ...", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    log.info("Done. Restart the inference container with MODEL_PATH=%s to serve it.",
             args.output_dir)


if __name__ == "__main__":
    main()
