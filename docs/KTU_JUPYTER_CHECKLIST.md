# KTU Jupyter/JupyterLab Checklist

Use this checklist during the first KTU session.

## First Checks

Run:

```bash
python scripts/check_ktu_environment.py
```

or open:

```text
notebooks/00_ktu_environment_check.ipynb
```

## What We Need To Confirm

1. GPU is available.
   - `nvidia-smi` works.
   - Python can see CUDA through PyTorch.

2. We can install packages.
   - `pip install fastapi uvicorn`
   - later: MusicGen dependencies.

3. We can run a REST server.
   - Start Uvicorn on a known port, e.g. `7860` or `8000`.
   - Check whether Jupyter exposes it through a proxy URL.

4. We can download model weights.
   - Hugging Face/Meta model downloads are allowed.
   - Cache should stay on the KTU server, not on the local laptop.

## Useful Commands

```bash
nvidia-smi
python --version
python -m pip --version
python -m pip list | head
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Jupyter Port Proxy Patterns To Try

Depending on the JupyterHub setup, one of these may work:

```text
https://<jupyter-host>/user/<username>/proxy/8000/
https://<jupyter-host>/proxy/8000/
```

If direct proxy paths are blocked, we can run both GUI and API inside Jupyter and access the GUI through the same proxy mechanism.

## REST Probe

Install the lightweight diagnostic requirements:

```bash
python -m pip install -r requirements-ktu-diagnostics.txt
```

Start a small test API:

```bash
python -m uvicorn scripts.probe_rest_api:app --host 0.0.0.0 --port 8000
```

Then try opening one of the proxy URLs ending in `/health`.
