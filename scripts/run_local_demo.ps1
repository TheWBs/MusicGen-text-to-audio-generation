$env:MUSICGEN_DEMO_MODE = "1"
$env:MUSICGEN_OUTPUT_DIR = "outputs"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
