from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_command(command: list[str]) -> dict:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": command,
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} was not found on PATH",
        }

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return {
        "command": command,
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def torch_status() -> dict:
    if not package_available("torch"):
        return {"installed": False}

    started_at = time.perf_counter()
    import torch

    status = {
        "installed": True,
        "version": torch.__version__,
        "import_seconds": round(time.perf_counter() - started_at, 2),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        status["current_device"] = torch.cuda.current_device()
        status["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        status["total_memory_gb"] = round(props.total_memory / 1024**3, 2)
    return status


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "working_directory": str(Path.cwd()),
        "repo_root": str(repo_root),
        "environment": {
            "jupyterhub_user": os.environ.get("JUPYTERHUB_USER"),
            "jupyter_server_root": os.environ.get("JUPYTER_SERVER_ROOT"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "commands": {
            "nvidia_smi": run_command(["nvidia-smi"]),
            "pip_version": run_command([sys.executable, "-m", "pip", "--version"]),
        },
        "packages": {
            "torch": torch_status(),
            "fastapi_installed": package_available("fastapi"),
            "uvicorn_installed": package_available("uvicorn"),
            "audiocraft_installed": package_available("audiocraft"),
            "transformers_installed": package_available("transformers"),
        },
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
