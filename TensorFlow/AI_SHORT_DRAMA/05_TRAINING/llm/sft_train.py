"""
LoRA/SFT fine-tuning for the shot-scripting LLM.
Input dataset: a JSONL file of {"instruction": ..., "response": ...} pairs
derived from structured story data (e.g. "write the next shot list for scene X"
-> the actual shot list text). Bring your own dataset; see README.md.

Example:
  python sft_train.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_path /path/to/instruction_response.jsonl \
    --output_dir ./out/qwen2.5-1.5b-shot-sft
"""
from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT training for the story/shot LLM")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_path", required=True, help="JSONL of instruction/response pairs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="comma-separated module names",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def formatting_func(example: dict) -> str:
    return PROMPT_TEMPLATE.format(instruction=example["instruction"], response=example["response"])


def main() -> None:
    args = parse_args()

    train_dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        task_type="CAUSAL_LM",
        bias="none",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        packing=args.packing,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
