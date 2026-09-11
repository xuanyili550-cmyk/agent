# Video model training config

This directory holds a config template (`config.yaml`) only; no training loop
is implemented here yet (video model fine-tuning stacks vary a lot by
backbone -- wire this up against whichever base model 06_MODELS/video ends up
using).

`config.yaml`'s `dataset.manifest_path` maps to
`04_DATASET/metadata/video_manifest.jsonl`, loaded via
`build_video_dataset` in `04_DATASET/dataset_loaders.py`. Note that
`video_manifest.jsonl`'s `video_path` column is left as a plain string (no
`datasets.Video` cast) since this repo ships no real video files.

Dataset clips must be supplied by the user, or use a Hugging Face dataset
with a clearly commercial-friendly license (e.g. CC0/MIT/Apache-2.0). Nothing
in this directory downloads or scrapes data.
