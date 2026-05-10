# MusicGen Text-to-Audio Demo

University project demonstrating text-to-audio generation with MusicGen through a FastAPI REST API and a minimal browser GUI.

## What It Does

- Generates short audio clips from text prompts.
- Provides a GUI that calls the backend through REST.
- Stores generated audio history.
- Includes a small fine-tuning workflow for a custom MusicCaps subset.

## Model

The project starts from `facebook/musicgen-small`, a pretrained MusicGen checkpoint available through Hugging Face Transformers.

For the final demo, the model was lightly fine-tuned on a small trumpet-focused subset from `CLAPv2/MusicCaps`. The fine-tuned checkpoint is expected at:

```text
checkpoints/musicgen-trumpet-demo
```

Large model checkpoints, datasets, generated WAV files, and cache files are intentionally excluded from git.

## REST API

Main endpoint used by the GUI:

```text
POST /api/generate
```

Other endpoints:

- `GET /api/health`
- `GET /api/history`
- `GET /api/audio/{filename}`

## Run Locally In Demo Mode

Local demo mode does not load MusicGen. It only generates a synthetic WAV, so the REST API and GUI can be tested on a weak machine.

```powershell
.\scripts\run_local_demo.ps1
```

Open:

```text
http://127.0.0.1:8000
```

## Run On KTU Jupyter GPU

Start the KTU Jupyter server with a PyTorch GPU environment.

Install dependencies:

```bash
bash scripts/install_ktu_requirements.sh
```

Run the fine-tuned model:

```bash
MUSICGEN_MODEL_ID=checkpoints/musicgen-trumpet-demo bash scripts/run_ktu_server.sh
```

Open through Jupyter proxy:

```text
https://ai-notebook.ktu.edu/user/<username>/proxy/8000/
```

## Fine-Tuning

Dataset format:

```text
datasets/custom_music/
  audio/
    trumpet_001.wav
    trumpet_002.wav
  metadata.csv
```

`metadata.csv` must contain:

```csv
file_name,text
audio/trumpet_001.wav,"caption describing the audio"
```

Stable fine-tuning command used on KTU:

```bash
python training/train_musicgen_decoder.py \
  --dataset-dir datasets/custom_music \
  --output-dir checkpoints/musicgen-trumpet-demo \
  --max-audio-seconds 4 \
  --max-steps 30 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-6 \
  --mixed-precision none
```

More details are in `docs/FINE_TUNING_PLAN.md`.
