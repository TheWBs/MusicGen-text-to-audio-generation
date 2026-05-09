# MusicGen Text-to-Audio Demo

University project for demonstrating how MusicGen-style text-to-audio generation works through a REST API and GUI.

The intended setup keeps heavy model execution on KTU Jupyter/JupyterLab servers:

- FastAPI backend runs on the remote GPU server.
- GUI calls the backend through REST endpoints.
- Generated audio files are saved and shown in history.
- Fine-tuning support is planned as an optional later extension.

Current status: working project scaffold with REST API, GUI, history, and pretrained MusicGen inference path.

## Run Locally In Demo Mode

Local demo mode does not load MusicGen. It generates a small synthetic WAV so the GUI and REST API can be tested on a weak laptop.

```powershell
.\scripts\run_local_demo.ps1
```

Open:

```text
http://127.0.0.1:8000
```

## Run On KTU Jupyter GPU

Start the Jupyter server with the `PyTorch` profile and enable `GPU enviroment`.

Install dependencies:

```bash
bash scripts/install_ktu_requirements.sh
```

Run the API and GUI:

```bash
bash scripts/run_ktu_server.sh
```

Expected GUI URL if `jupyter-server-proxy` is available:

```text
https://ai-notebook.ktu.edu/user/jokgri@ktu.lt/proxy/8000/
```

## REST API

- `GET /api/health`
- `POST /api/generate`
- `GET /api/history`
- `GET /api/audio/{filename}`

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"80s synthwave track with bright arpeggios","duration_seconds":8}'
```

## Notes

- Do not commit generated audio or model weights.
- Model/cache files should stay on KTU Jupyter storage.
- Fine-tuning plan: `docs/FINE_TUNING_PLAN.md`.
