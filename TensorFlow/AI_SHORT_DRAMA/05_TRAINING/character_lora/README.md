# Character LoRA training (SDXL)

`train_character_lora_sdxl.py` trains a per-character DreamBooth-style LoRA on
the SDXL UNet, following the structure of diffusers' `dreambooth_lora_sdxl`
example script. Input dataset is `04_DATASET/metadata/character_manifest.jsonl`
(loaded through `04_DATASET/dataset_loaders.build_character_dataset`),
filtered down to one `character_id` per run.

Optional prior-preservation (`--with_prior_preservation`) expects a
user-supplied folder of class images (`--class_data_dir`) -- this script never
generates or downloads class images itself.

Run:
```
python train_character_lora_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --manifest_path ../../04_DATASET/metadata/character_manifest.jsonl \
  --character_id char_lin_wei \
  --instance_prompt "a photo of sks_lin_wei person" \
  --output_dir ./out/char_lin_wei_lora
```

Dataset images must be supplied by the user, or use a Hugging Face dataset
with a clearly commercial-friendly license (e.g. CC0/MIT/Apache-2.0). This
script does not download or scrape any data itself. The placeholder PNGs
under `04_DATASET/characters/` are synthetic smoke-test data only, not real
training material.
