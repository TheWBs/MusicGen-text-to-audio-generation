from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, MusicgenForConditionalGeneration


@dataclass
class TrainingSummary:
    base_model: str
    dataset_dir: str
    output_dir: str
    device: str
    sample_rate: int
    train_examples: int
    train_seconds_per_example: float
    max_steps: int
    learning_rate: float
    trainable_parameters: int
    total_parameters: int
    initial_loss: float | None
    final_loss: float | None
    best_loss: float | None
    loss_delta: float | None
    runtime_seconds: float


class CaptionedAudioDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset_dir: Path,
        sample_rate: int,
        max_audio_seconds: float,
        limit: int | None = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_audio_seconds)

        metadata_path = dataset_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing {metadata_path}")

        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        if limit is not None:
            rows = rows[:limit]

        self.rows = rows
        if not self.rows:
            raise ValueError("Dataset is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        file_name = row["file_name"]
        caption = row["text"]
        audio_path = self.dataset_dir / file_name
        audio, input_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if int(input_rate) != self.sample_rate:
            divisor = math.gcd(int(input_rate), self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // divisor, int(input_rate) // divisor)
            audio = np.asarray(audio, dtype=np.float32)

        if audio.shape[0] > self.max_samples:
            audio = audio[: self.max_samples]
        elif audio.shape[0] < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - audio.shape[0]))

        return {
            "text": caption,
            "audio": audio,
            "file_name": file_name,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def collate_batch(
    rows: list[dict[str, Any]],
    processor: Any,
    sample_rate: int,
) -> dict[str, Any]:
    text_inputs = processor(
        text=[row["text"] for row in rows],
        padding=True,
        return_tensors="pt",
    )
    audio_inputs = processor(
        audio=[row["audio"] for row in rows],
        sampling_rate=sample_rate,
        padding=True,
        return_tensors="pt",
    )
    return {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "input_values": audio_inputs["input_values"],
        "padding_mask": audio_inputs.get("padding_mask"),
        "file_names": [row["file_name"] for row in rows],
    }


def encode_audio_codes(
    model: MusicgenForConditionalGeneration,
    input_values: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    audio_encoder_kwargs = {"input_values": input_values}
    if padding_mask is not None:
        audio_encoder_kwargs["padding_mask"] = padding_mask

    with torch.no_grad():
        audio_encoder_outputs = model.audio_encoder.encode(**audio_encoder_kwargs)

    audio_codes = audio_encoder_outputs.audio_codes
    if audio_codes.dim() == 4:
        audio_codes = audio_codes[0]

    if model.config.audio_encoder.audio_channels == 2 and audio_codes.shape[2] == model.decoder.num_codebooks // 2:
        audio_codes = audio_codes.transpose(1, 2).reshape(
            -1,
            model.decoder.num_codebooks,
            audio_codes.shape[-1],
        )

    return audio_codes.transpose(1, 2).contiguous()


def train(args: argparse.Namespace) -> TrainingSummary:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float16 if device.type == "cuda" and args.mixed_precision == "fp16" else torch.float32

    processor = AutoProcessor.from_pretrained(args.base_model)
    model = MusicgenForConditionalGeneration.from_pretrained(args.base_model, torch_dtype=dtype)
    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.decoder.parameters():
        parameter.requires_grad = True

    model.text_encoder.eval()
    model.audio_encoder.eval()
    model.decoder.train()
    model.config.use_cache = False
    if hasattr(model.decoder.config, "use_cache"):
        model.decoder.config.use_cache = False
    if getattr(model.config.decoder, "decoder_start_token_id", None) is None:
        model.config.decoder.decoder_start_token_id = (
            model.decoder.config.bos_token_id
            if model.decoder.config.bos_token_id is not None
            else model.decoder.config.pad_token_id
        )
    if getattr(model.config.decoder, "pad_token_id", None) is None:
        model.config.decoder.pad_token_id = model.decoder.config.pad_token_id

    sample_rate = int(model.config.audio_encoder.sampling_rate)
    dataset = CaptionedAudioDataset(
        dataset_dir=Path(args.dataset_dir),
        sample_rate=sample_rate,
        max_audio_seconds=args.max_audio_seconds,
        limit=args.limit_examples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_batch(rows, processor, sample_rate),
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    trainable_parameters, total_parameters = count_parameters(model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    losses: list[float] = []
    started_at = time.perf_counter()
    step = 0
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_values = batch["input_values"].to(device=device, dtype=dtype)
            padding_mask = batch["padding_mask"]
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            labels = encode_audio_codes(model, input_values, padding_mask)
            labels = labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            actual_loss = float(loss.detach().cpu().item() * args.gradient_accumulation_steps)
            losses.append(actual_loss)
            step += 1
            print(f"step={step:04d} loss={actual_loss:.4f} file={batch['file_names'][0]}", flush=True)

            if step >= args.max_steps:
                break

    if step % args.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    initial_loss = losses[0] if losses else None
    final_loss = losses[-1] if losses else None
    loss_delta = None if initial_loss is None or final_loss is None else initial_loss - final_loss
    summary = TrainingSummary(
        base_model=args.base_model,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        device=str(device),
        sample_rate=sample_rate,
        train_examples=len(dataset),
        train_seconds_per_example=args.max_audio_seconds,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=min(losses) if losses else None,
        loss_delta=loss_delta,
        runtime_seconds=round(time.perf_counter() - started_at, 2),
    )

    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2)

    print()
    print(f"Saved checkpoint to {output_dir}")
    print(f"Saved summary to {summary_path}")
    if initial_loss is not None and final_loss is not None:
        print(f"Loss: {initial_loss:.4f} -> {final_loss:.4f}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MusicGen decoder fine-tuning script.")
    parser.add_argument("--base-model", default="facebook/musicgen-small")
    parser.add_argument("--dataset-dir", default="datasets/custom_music")
    parser.add_argument("--output-dir", default="checkpoints/musicgen-trumpet-demo")
    parser.add_argument("--max-audio-seconds", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=["none", "fp16"], default="fp16")
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
