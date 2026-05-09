# Fine-Tuning Plan

The first deliverable uses a pretrained MusicGen checkpoint. Fine-tuning is intentionally separated because it needs a prepared audio dataset and a longer GPU run.

## Dataset Format

Recommended project-local layout:

```text
datasets/custom_music/
  audio/
    track_001.wav
    track_002.wav
  metadata.csv
```

`metadata.csv`:

```csv
file_name,text
audio/track_001.wav,"short cinematic piano phrase with soft reverb"
audio/track_002.wav,"fast electronic beat with bright synth lead"
```

## Practical Strategy

MusicGen's text encoder and audio codec are frozen in the original architecture. For a realistic university extension, fine-tuning should focus on the decoder / continuation training rather than training from scratch.

Implemented scripts:

- `training/prepare_dataset.py` - validate audio, resample to 32 kHz if needed, check captions.
- `training/train_musicgen_decoder.py` - continue training from `facebook/musicgen-small`.
- `training/config.example.yaml` - small-batch configuration for KTU GPU.

Stable KTU command used for the MusicCaps trumpet subset:

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

FP16 training without gradient scaling produced `nan` loss after a few steps in the KTU environment, so the stable demo command uses float32.

## Minimal Acceptance Path

1. Prepare 10-30 short WAV files.
2. Write captions that describe instrumentation, mood and tempo.
3. Run a short continuation training job on KTU GPU.
4. Save a checkpoint under `checkpoints/`.
5. Start the API with `MUSICGEN_MODEL_ID=checkpoints/<checkpoint-name>`.

This keeps the pretrained demo working while leaving a clear path to extra training later.
