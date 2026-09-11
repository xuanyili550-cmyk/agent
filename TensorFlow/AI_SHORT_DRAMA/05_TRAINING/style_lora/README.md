# Style LoRA training (SDXL)

`train_style_lora_sdxl.py` trains a visual-style LoRA on the SDXL UNet, using
`04_DATASET/metadata/style_manifest.jsonl` (loaded through
`04_DATASET/dataset_loaders.build_style_dataset`), filtered down to one
`style_id` per run. No prior-preservation class images -- style LoRA is not a
subject-identity DreamBooth run.

Run:
```
python train_style_lora_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --manifest_path ../../04_DATASET/metadata/style_manifest.jsonl \
  --style_id style_cinematic_moody \
  --style_trigger_token "in sks_cinematic_moody style" \
  --output_dir ./out/style_cinematic_moody_lora
```

Dataset images must be supplied by the user, or use a Hugging Face dataset
with a clearly commercial-friendly license (e.g. CC0/MIT/Apache-2.0). This
script does not download or scrape any data itself. The placeholder PNGs
under `04_DATASET/styles/` are synthetic smoke-test data only, not real
training material.
