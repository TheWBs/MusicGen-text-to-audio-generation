from __future__ import annotations

import os
import socket
from datetime import datetime, timezone

from fastapi import FastAPI


app = FastAPI(title="MusicGen REST API Probe")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "musicgen-rest-probe",
        "host": socket.gethostname(),
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "jupyterhub_user": os.environ.get("JUPYTERHUB_USER"),
    }


@app.get("/")
def root() -> dict:
    return {
        "message": "REST probe is running. Try /health.",
    }
