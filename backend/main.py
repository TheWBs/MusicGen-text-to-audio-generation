from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .history import HistoryItem, append_history, ensure_output_dir, load_history, new_audio_filename
from .musicgen_service import GenerationRequest, service


class GeneratePayload(BaseModel):
    prompt: Annotated[str, Field(min_length=3, max_length=500)]
    duration_seconds: Annotated[int, Field(ge=2, le=settings.max_duration_seconds)] = 8
    guidance_scale: Annotated[float, Field(ge=1.0, le=10.0)] = 3.0
    temperature: Annotated[float, Field(ge=0.1, le=2.0)] = 1.0
    seed: Annotated[int | None, Field(ge=0, le=2_147_483_647)] = None


app = FastAPI(title="MusicGen REST API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    ensure_output_dir()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_id": settings.model_id,
        "model_loaded": service.loaded,
        "device": service.device,
        "demo_mode": settings.demo_mode,
        "max_duration_seconds": settings.max_duration_seconds,
    }


@app.get("/api/history")
def history() -> list[dict]:
    return load_history()


@app.post("/api/generate")
def generate(payload: GeneratePayload) -> dict:
    filename = new_audio_filename()
    output_path = settings.output_dir / filename

    try:
        result = service.generate(
            GenerationRequest(
                prompt=payload.prompt.strip(),
                duration_seconds=payload.duration_seconds,
                guidance_scale=payload.guidance_scale,
                temperature=payload.temperature,
                seed=payload.seed,
            ),
            output_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    item = HistoryItem(
        id=Path(filename).stem,
        prompt=payload.prompt.strip(),
        model_id=result.model_id,
        duration_seconds=payload.duration_seconds,
        guidance_scale=payload.guidance_scale,
        temperature=payload.temperature,
        seed=payload.seed,
        filename=filename,
        audio_url=f"api/audio/{filename}",
        created_at=datetime.now(timezone.utc).isoformat(),
        generation_seconds=result.generation_seconds,
        device=result.device,
        demo_mode=result.demo_mode,
    )
    return append_history(item)


@app.get("/api/audio/{filename}")
def audio(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = settings.output_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)


static_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
