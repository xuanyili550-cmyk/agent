"""
分镜 / 剧本 LLM 的 LoRA SFT（监督微调）训练脚本。

在流水线中的位置：05_TRAINING/llm 的第二步。输入是 ``build_sft_dataset.py`` 产出的 JSONL
（每行 {"instruction": ..., "response": ...}，instruction 就是故事引擎各 agent 实际发出的 user prompt，
response 是通过 schema 校验的 JSON），训练完的 checkpoint 可直接替换 ``LocalTransformersProvider`` 的底座模型，
prompt 格式零改动。训练效果用 ``evaluate_structured_output.py`` 量化。

为什么用 LoRA 而不是全参微调：目标只是让底座模型稳定输出符合项目 schema 的 JSON，这属于"格式与领域习惯"的适配，
LoRA 的低秩增量足够，且显存占用小（单卡即可训 1.5B~7B 模型）、产物只有几十 MB、便于按项目 / 题材切换适配器。

示例：
  python sft_train.py \\
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \\
    --dataset_path /path/to/instruction_response.jsonl \\
    --output_dir ./out/qwen2.5-1.5b-shot-sft
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Alpaca 风格的 instruction/response 模板；训练与推理必须使用同一模板，否则模型学到的格式对不上
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    关键默认值的取舍：
    - ``lora_r=16, lora_alpha=32``：alpha/r = 2 是社区常用的缩放比；r=16 对"学会输出固定 JSON 结构"这种任务已足够，
      再大收益不明显但会增大产物与过拟合风险。
    - ``lora_target_modules`` 同时覆盖注意力（q/k/v/o）与 MLP（gate/up/down）投影：只挂注意力层对格式类任务收敛偏慢。
    - ``learning_rate=2e-4``：LoRA 参数量小，可以比全参微调（通常 1e-5 量级）高一个数量级。
    - ``batch=1 x grad_accum=8``：等效 batch 8，兼顾单卡显存与梯度稳定性。
    """
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
    """把一条 instruction/response 样本渲染成训练文本（SFTTrainer 会对整段文本做因果语言建模）。"""
    return PROMPT_TEMPLATE.format(instruction=example["instruction"], response=example["response"])


def main() -> None:
    """训练主流程：加载数据与底座模型 -> 配置 LoRA -> 用 TRL 的 SFTTrainer 训练 -> 保存适配器与 tokenizer。"""
    args = parse_args()

    train_dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # 不少因果 LM（如 Qwen / Llama）没有 pad token，批处理时需要一个；借用 eos 是通用做法
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        task_type="CAUSAL_LM",
        bias="none",  # 不训练 bias，进一步减小可训练参数量
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

    # 把 peft_config 交给 SFTTrainer，由它负责把 LoRA 挂到模型上，避免手动 get_peft_model 与 trainer 内部逻辑重复
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        formatting_func=formatting_func,
    )

    trainer.train()
    # 只保存 LoRA 适配器权重（不含底座），并把 tokenizer 一并存下，推理侧可直接 from_pretrained
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
