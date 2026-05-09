# MusicGen Text-to-Audio Demo

University project for demonstrating how MusicGen-style text-to-audio generation works through a REST API and GUI.

The intended setup keeps heavy model execution on KTU Jupyter/JupyterLab servers:

- FastAPI backend runs on the remote GPU server.
- GUI calls the backend through REST endpoints.
- Generated audio files are saved and shown in history.
- Fine-tuning support is planned as an optional later extension.

Current status: planning and KTU environment diagnostics.

## Next Step

Open `notebooks/00_ktu_environment_check.ipynb` or run `scripts/check_ktu_environment.py` inside KTU JupyterLab to verify:

- GPU/CUDA availability
- Python and package versions
- whether a FastAPI port can be exposed through Jupyter
- whether Hugging Face / MusicGen dependencies can be installed

