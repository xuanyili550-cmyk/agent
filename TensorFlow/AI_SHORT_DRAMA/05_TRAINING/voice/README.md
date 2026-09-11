# Voice / TTS model training config

This directory holds a config template (`config.yaml`) only; no training loop
is implemented here yet.

`config.yaml`'s `dataset.manifest_path` maps to
`04_DATASET/metadata/voice_manifest.jsonl`, loaded via `build_voice_dataset`
in `04_DATASET/dataset_loaders.py`. `audio_path` is kept as a plain string
column (no `datasets.Audio` cast) since this repo ships no real audio files.

Dataset audio must be supplied by the user, or use a Hugging Face dataset
with a clearly commercial-friendly license (e.g. CC0/MIT/Apache-2.0). Nothing
in this directory downloads or scrapes data.
