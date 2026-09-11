# LLM SFT / LoRA training

`sft_train.py` fine-tunes a causal LM with `trl.SFTTrainer` + a PEFT LoRA
adapter, on an instruction/response JSONL dataset (e.g. shot-list generation
conditioned on story structure).

Dataset format (one JSON object per line):
```json
{"instruction": "Write shot SH003 for scene EP001_SC01 ...", "response": "Medium shot, ..."}
```
This does not correspond to a `04_DATASET/metadata` manifest directly -- it is
derived from `03_STRUCTURED_DATA` story/shot JSON by a separate data-prep step
not included here.

Run:
```
python sft_train.py \
  --model_name_or_path <hf-model-id-or-local-path> \
  --dataset_path /path/to/instruction_response.jsonl \
  --output_dir ./out/run1
```

Dataset must be supplied by the user, or use a Hugging Face dataset with a
clearly commercial-friendly license (e.g. CC0/MIT/Apache-2.0). This script
does not download or scrape any data itself.
