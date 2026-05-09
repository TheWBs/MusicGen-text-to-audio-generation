from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_id: str = os.environ.get("MUSICGEN_MODEL_ID", "facebook/musicgen-small")
    device: str = os.environ.get("MUSICGEN_DEVICE", "auto")
    output_dir: Path = Path(os.environ.get("MUSICGEN_OUTPUT_DIR", "outputs")).resolve()
    history_file: Path = Path(os.environ.get("MUSICGEN_HISTORY_FILE", "outputs/history.json")).resolve()
    demo_mode: bool = env_bool("MUSICGEN_DEMO_MODE", False)
    max_duration_seconds: int = int(os.environ.get("MUSICGEN_MAX_DURATION_SECONDS", "20"))


settings = Settings()
