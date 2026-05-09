from __future__ import annotations

import math
import random
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from .config import settings


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    duration_seconds: int
    guidance_scale: float
    temperature: float
    seed: int | None


@dataclass(frozen=True)
class GenerationResult:
    path: Path
    generation_seconds: float
    device: str
    model_id: str
    demo_mode: bool


class MusicGenService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._processor = None
        self._model = None
        self._torch = None
        self._device = "stub" if settings.demo_mode else "not-loaded"

    @property
    def device(self) -> str:
        return self._device

    @property
    def loaded(self) -> bool:
        return settings.demo_mode or self._model is not None

    def generate(self, request: GenerationRequest, output_path: Path) -> GenerationResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()

        with self._lock:
            if settings.demo_mode:
                self._write_stub_audio(request, output_path)
            else:
                self._ensure_model_loaded()
                self._generate_with_transformers(request, output_path)

        return GenerationResult(
            path=output_path,
            generation_seconds=round(time.perf_counter() - started_at, 2),
            device=self._device,
            model_id=settings.model_id,
            demo_mode=settings.demo_mode,
        )

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        if settings.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = settings.device

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self._processor = AutoProcessor.from_pretrained(settings.model_id)
        self._model = MusicgenForConditionalGeneration.from_pretrained(
            settings.model_id,
            torch_dtype=dtype,
        )
        self._model.to(device)
        self._model.eval()
        self._torch = torch
        self._device = device

    def _generate_with_transformers(self, request: GenerationRequest, output_path: Path) -> None:
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        torch = self._torch
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(request.seed)

        inputs = self._processor(
            text=[request.prompt],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        # MusicGen produces roughly 50 audio tokens per second.
        max_new_tokens = max(32, min(request.duration_seconds, settings.max_duration_seconds) * 50)

        with torch.inference_mode():
            audio_values = self._model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=request.guidance_scale,
                temperature=request.temperature,
                max_new_tokens=max_new_tokens,
                generator=generator,
            )

        audio = audio_values[0, 0].detach().float().cpu().numpy()
        sampling_rate = int(self._model.config.audio_encoder.sampling_rate)

        from scipy.io.wavfile import write

        audio = np.clip(audio, -1.0, 1.0)
        write(output_path, sampling_rate, audio)

    def _write_stub_audio(self, request: GenerationRequest, output_path: Path) -> None:
        seed = request.seed if request.seed is not None else random.randint(0, 2**31 - 1)
        rng = random.Random(seed)
        sample_rate = 32000
        seconds = min(request.duration_seconds, settings.max_duration_seconds)
        frames = sample_rate * seconds
        base = 180 + rng.randint(0, 180)

        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            for i in range(frames):
                t = i / sample_rate
                envelope = min(1.0, i / (sample_rate * 0.2), (frames - i) / (sample_rate * 0.35))
                tone = math.sin(2 * math.pi * base * t)
                overtone = 0.35 * math.sin(2 * math.pi * (base * 1.5) * t)
                pulse = 0.18 * math.sin(2 * math.pi * 3.0 * t)
                value = int(16000 * envelope * (tone + overtone) * (0.8 + pulse))
                wav.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))


service = MusicGenService()
