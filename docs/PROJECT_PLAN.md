# Project Plan

## Goal

Build a MusicGen text-to-audio demo that satisfies the university project requirements:

- A GUI for live demonstration.
- GUI communicates with a REST API.
- Backend generates audio from text prompts using a pretrained MusicGen model.
- Generated files are kept in a history list.
- The codebase leaves a path for later fine-tuning on a custom dataset.

## Proposed Architecture

```text
Browser GUI
   |
   | HTTP REST calls
   v
FastAPI backend on KTU Jupyter server
   |
   | loads pretrained MusicGen model
   v
GPU inference + generated audio files
```

## Components

### Backend

Technology:

- Python
- FastAPI
- Uvicorn
- MusicGen implementation, likely through Meta Audiocraft or Hugging Face Transformers depending on KTU compatibility

Core endpoints:

- `GET /health` - confirms backend is alive and reports device/model status.
- `POST /generate` - accepts prompt, duration and optional generation settings; returns generated audio metadata.
- `GET /audio/{filename}` - serves generated WAV files.
- `GET /history` - returns previous generations.
- Later optional: `POST /fine-tune` or a separate fine-tuning script/notebook.

### GUI

Technology options:

- Preferred: React + Vite for a polished demo interface.
- Fallback: simple static HTML/CSS/JS if KTU/server constraints make React inconvenient.

Expected GUI features:

- Prompt input.
- Duration selector.
- Generate button with loading state.
- Audio player for the latest result.
- History of previous generations.
- Backend URL configuration if GUI runs locally and API runs remotely.

### Fine-Tuning Extension

Planned but not required for the first working demo:

- Dataset folder with audio files.
- Metadata file mapping audio files to text captions.
- Training config template.
- Script/notebook for lightweight fine-tuning or adapter-style continuation if the selected MusicGen stack supports it in the KTU environment.

## Implementation Phases

1. KTU environment verification.
   - Check GPU and CUDA.
   - Check Python version.
   - Check package installation permissions.
   - Check outbound access to model weights.
   - Check whether REST ports can be reached from the browser.

2. Minimal backend.
   - FastAPI app.
   - Health endpoint.
   - Temporary stub generation endpoint for GUI integration.

3. Pretrained MusicGen generation.
   - Load a small pretrained model first.
   - Generate short WAV files.
   - Save generated outputs and metadata.

4. GUI.
   - Build demonstration interface.
   - Connect it only through REST API.
   - Add history view and audio playback.

5. Jupyter deployment workflow.
   - Clear commands for installing dependencies.
   - Clear commands for starting backend.
   - Document the correct URL/proxy path for the GUI to call the API.

6. Optional fine-tuning support.
   - Dataset preparation guide.
   - Training script or notebook.
   - Minimal example config.

## Main Risks

- KTU Jupyter may not expose arbitrary ports directly.
- MusicGen dependency stack may require a Python version different from the default environment.
- Fine-tuning may need significantly more GPU memory than inference.
- Model downloads may be blocked or slow.

## Current Decision

Keep the local machine lightweight. Do not download models locally. Use the local repo for source code, documentation and GUI development only.

