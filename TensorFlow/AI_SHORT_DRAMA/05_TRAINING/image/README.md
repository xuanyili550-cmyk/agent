# Image model training config

This directory holds a config template (`config.yaml`) for image-model
fine-tuning runs, not a training script. For runnable scripts see:
- `05_TRAINING/character_lora/train_character_lora_sdxl.py` (per-character LoRA)
- `05_TRAINING/style_lora/train_style_lora_sdxl.py` (per-style LoRA)

`config.yaml`'s `dataset.manifest_path` maps to one of the
`04_DATASET/metadata/*.jsonl` manifests:
- `character_manifest.jsonl` -> `build_character_dataset`
- `scene_manifest.jsonl` -> `build_scene_dataset`
- `style_manifest.jsonl` -> `build_style_dataset`

all defined in `04_DATASET/dataset_loaders.py`.

Dataset images must be supplied by the user, or use a Hugging Face dataset
with a clearly commercial-friendly license (e.g. CC0/MIT/Apache-2.0). Nothing
in this directory downloads or scrapes data.
