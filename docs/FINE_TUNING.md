# Fine-Tuning Summary

This project includes a minimal continuation-training run for `facebook/musicgen-small`.

## Dataset

The dataset was created from `CLAPv2/MusicCaps` by selecting rows where either `caption` or `aspect_list` contains `trumpet`.

```text
datasets/custom_music/
  audio/
    trumpet_001.wav
    ...
    trumpet_030.wav
  metadata.csv
  subset_manifest.csv
```

`metadata.csv` format:

```csv
file_name,text
audio/trumpet_001.wav,"caption describing the audio"
```

Dataset statistics:

- Audio files: 30
- Dataset size: approximately 55 MB
- Audio sample rate used by MusicGen: 32 kHz
- Caption source: MusicCaps text metadata

## Training Setup

Base checkpoint:

```text
facebook/musicgen-small
```

Training command:

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

The text encoder and audio encoder were frozen. The decoder was trained on the captioned audio tokens.

## Results

Training statistics:

- Total parameters: 586,884,674
- Trainable parameters: 419,584,000
- Frozen parameters: approximately 167.3M
- Initial loss: 9.3252
- Final loss: 8.4840
- Best loss: 8.3585
- Checkpoint size: approximately 2.2 GB

The fine-tuned checkpoint path is:

```text
checkpoints/musicgen-trumpet-demo
```
