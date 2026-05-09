from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from .config import settings


@dataclass
class HistoryItem:
    id: str
    prompt: str
    model_id: str
    duration_seconds: int
    guidance_scale: float
    temperature: float
    seed: int | None
    filename: str
    audio_url: str
    created_at: str
    generation_seconds: float
    device: str
    demo_mode: bool


_history_lock = Lock()


def ensure_output_dir() -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.history_file.parent.mkdir(parents=True, exist_ok=True)


def new_audio_filename() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid4().hex[:8]}.wav"


def load_history() -> list[dict[str, Any]]:
    ensure_output_dir()
    if not settings.history_file.exists():
        return []
    try:
        return json.loads(settings.history_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_history(items: list[dict[str, Any]]) -> None:
    ensure_output_dir()
    settings.history_file.write_text(json.dumps(items, indent=2), encoding="utf-8")


def append_history(item: HistoryItem) -> dict[str, Any]:
    with _history_lock:
        items = load_history()
        record = asdict(item)
        items.insert(0, record)
        save_history(items[:100])
        return record
