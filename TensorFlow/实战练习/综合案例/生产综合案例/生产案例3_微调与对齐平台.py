"""
================================================================================
 生产综合案例3 · 模型微调与对齐平台（真实【生产 GPU 训练】，本机不跑）
================================================================================
 ⚠ 这三条训练线都是【生产 GPU 训练】的真实代码：用官方 trl.Trainer/SFTTrainer/GRPOTrainer +
   真实数据集 + bf16/梯度检查点/多卡(Accelerate/DeepSpeed) + 推模型仓库。需 GPU 云环境，本机跑不了。
   (不是本地玩具小循环——按用户要求，生产训练就按 GPU 生产方式写。)
   本地能跑的小版本见 ../../chapter实战/分章项目/Ch2/Ch11/Ch12；本文件的 GRPO【原理张量自检】可本地跑。
 覆盖章节功能(生产训练方式)：
   [Ch2 微调]   transformers Trainer + 真实 ag_news + 生产 TrainingArguments + compute_metrics + push_to_hub
   [Ch11 LoRA]  trl.SFTTrainer + SFTConfig + LoRA + 真实指令数据 + bf16/梯度检查点 + push_to_hub
   [Ch12 GRPO]  trl.GRPOTrainer + GRPOConfig + use_vllm 加速采样 + 真实 GSM8K + 可验证奖励 + LoRA
   [Ch9 Gradio] 训练平台 Web(在 GPU 机上点按钮触发训练/看指标)

 —— 多卡启动(Accelerate/DeepSpeed，在 GPU 机) ——
   accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 生产案例3_微调与对齐平台.py --train ch2
   # 或写 accelerate_config.yaml 配 DeepSpeed ZeRO-2/3(见 ../../chapter实战/生产架构/生产03)
 运行(本机)：
   python3 生产案例3_微调与对齐平台.py smoke   # GRPO 原理张量自检(CPU可跑) + 校验三条训练函数已定义
   python3 生产案例3_微调与对齐平台.py         # 起训练平台 Web(训练按钮需 GPU)
================================================================================
"""
import sys
import re
import gradio as gr


# ==============================================================================
# [Ch2] 文本分类微调 —— transformers Trainer 生产写法(真实数据 + 生产参数 + 推仓库)
# ==============================================================================
def train_classifier(push_to="your-org/ag-news-distilbert"):
    """生产 GPU 训练：ag_news 全量微调 distilbert，bf16 + 每轮评估 + 最优检查点 + 推模型仓库。"""
    import numpy as np
    from datasets import load_dataset                          # 生产 datasets 正常
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainingArguments)
    ckpt = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(ckpt)
    ds = load_dataset("fancyzhx/ag_news")                      # 真实全量数据
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)

    def metrics(p):
        preds = p.predictions.argmax(-1)
        return {"accuracy": float((preds == p.label_ids).mean())}

    args = TrainingArguments(
        output_dir="/mnt/ckpt/ag-news", num_train_epochs=3,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        learning_rate=3e-5, warmup_ratio=0.1, weight_decay=0.01,
        bf16=True, gradient_checkpointing=True,                # 生产：混合精度 + 省显存
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        logging_steps=50, report_to="wandb",
        push_to_hub=True, hub_model_id=push_to,                # 训完推模型仓库(推理服务从这拉)
    )
    trainer = Trainer(model=AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=4),
                      args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
                      data_collator=DataCollatorWithPadding(tokenizer=tok), compute_metrics=metrics)
    trainer.train()
    trainer.push_to_hub()
    return trainer


# ==============================================================================
# [Ch11] LoRA 指令微调 —— trl.SFTTrainer 生产写法(真实指令数据 + LoRA + bf16 + 推仓库)
# ==============================================================================
def train_lora_sft(model_id="Qwen/Qwen2.5-7B-Instruct", push_to="your-org/qwen-sft-lora"):
    """生产 GPU 训练：trl.SFTTrainer + LoRA + DeepSpeed(由 accelerate 启动)。"""
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")   # 真实指令数据
    lora = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                      lora_dropout=0.05, task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir="/mnt/ckpt/qwen-sft", num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,      # 有效 batch=4*4*卡数
        learning_rate=2e-4, bf16=True, gradient_checkpointing=True,
        packing=True, max_length=1024, logging_steps=10, save_steps=500,
        report_to="wandb", push_to_hub=True, hub_model_id=push_to,
    )
    trainer = SFTTrainer(model=model_id, args=args, train_dataset=ds, peft_config=lora)
    trainer.train(); trainer.push_to_hub()
    return trainer


# ==============================================================================
# [Ch12] GRPO 强化 —— trl.GRPOTrainer 生产写法(vLLM 加速采样 + 真实 GSM8K + 可验证奖励 + LoRA)
# ==============================================================================
def _extract(text):
    return text.split("####")[-1].strip() if "####" in text else text


def correctness_reward(completions, answer, **kw):
    return [2.0 if _extract(c) == a else 0.0 for c, a in zip(completions, answer)]


def format_reward(completions, **kw):
    pat = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    return [0.5 if re.search(pat, c, re.DOTALL) else 0.0 for c in completions]


def train_grpo(model_id="Qwen/Qwen2.5-7B-Instruct"):
    """生产 GPU 训练：trl.GRPOTrainer + vLLM 加速采样 + LoRA + 可验证奖励。"""
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    dataset = load_dataset("openai/gsm8k", "main", split="train")          # 真实数学题(可验证)
    args = GRPOConfig(
        output_dir="/mnt/ckpt/grpo-qwen", learning_rate=5e-6,
        per_device_train_batch_size=8, gradient_accumulation_steps=4,
        num_generations=8,                    # ★每题采一组 8 个
        max_prompt_length=256, max_completion_length=1024,
        use_vllm=True,                        # ★vLLM 加速采样(生产必开)
        bf16=True, max_steps=1000, logging_steps=1, report_to="wandb",
    )
    trainer = GRPOTrainer(model=model_id, reward_funcs=[correctness_reward, format_reward],
                          args=args, train_dataset=dataset,
                          peft_config=LoraConfig(r=16, lora_alpha=32,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    trainer.train(); trainer.save_model("/mnt/ckpt/grpo-final")
    return trainer


# ==============================================================================
# [Ch12] GRPO 原理张量自检(纯张量，本地 CPU 可跑，证明你懂算法底层)
# ==============================================================================
def grpo_principle():
    import torch
    rewards = torch.tensor([1., 0., 0., 1., 0., 0., 1., 1.]); G = 4
    g = rewards.view(-1, G)
    adv = (rewards - g.mean(1).repeat_interleave(G)) / (g.std(1).repeat_interleave(G) + 1e-8)
    assert format_reward(["<reasoning>x</reasoning><answer>5</answer>", "no"]) == [0.5, 0.0]
    assert correctness_reward(["#### 5", "#### 4"], ["5", "5"]) == [2.0, 0.0]

    def loss(new, old, ref, adv, eps=0.2, beta=0.04):
        ratio = torch.exp(new - old); a = adv.unsqueeze(1)
        pol = -torch.min(ratio * a, torch.clamp(ratio, 1 - eps, 1 + eps) * a)
        kl = torch.exp(ref - new) - (ref - new) - 1
        return (pol + beta * kl).mean()
    torch.manual_seed(0)
    o = torch.randn(8, 1); n = o + 0.1 * torch.randn(8, 1)
    return (f"[Ch12] GRPO 原理自检：组内优势={[round(x,2) for x in adv.tolist()][:4]}...；"
            f"奖励函数OK；裁剪损失+KL={loss(n, o, o.clone(), adv).item():.4f}")


def _need_gpu(fn):
    def wrap():
        try:
            fn()
            return "训练已启动(需在 GPU 机上运行)"
        except Exception as e:
            return f"⚠ 生产 GPU 训练，本机跑不了：{type(e).__name__}: {str(e)[:80]}\n请在 GPU 云上 `accelerate launch ...`"
    return wrap


def build_demo():
    with gr.Blocks(title="微调与对齐平台", analytics_enabled=False) as demo:
        gr.Markdown("# 模型微调与对齐平台（生产 GPU 训练）\n"
                    "三条训练线都是生产写法(trl Trainer/SFTTrainer/GRPOTrainer)，在 GPU 机上触发。")
        with gr.Tab("文本分类微调(Ch2)"):
            o1 = gr.Textbox(label="状态", lines=2)
            gr.Button("启动生产训练(需GPU)", variant="primary").click(_need_gpu(train_classifier), None, o1)
        with gr.Tab("LoRA 指令微调(Ch11)"):
            o2 = gr.Textbox(label="状态", lines=2)
            gr.Button("启动生产训练(需GPU)", variant="primary").click(_need_gpu(train_lora_sft), None, o2)
        with gr.Tab("GRPO 强化(Ch12)"):
            o3 = gr.Textbox(label="状态", lines=2)
            gr.Button("启动生产训练(需GPU+vLLM)", variant="primary").click(_need_gpu(train_grpo), None, o3)
            o4 = gr.Textbox(label="GRPO 原理自检(本地可跑)", lines=2)
            gr.Button("跑 GRPO 原理张量自检", variant="secondary").click(grpo_principle, None, o4)
    return demo


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        grpo_principle()                                     # 原理自检(CPU 可跑，内含 assert)，返回串输出已省
        for fn in (train_classifier, train_lora_sft, train_grpo):
            assert callable(fn)                              # 生产训练函数已定义(真实 GPU 代码，不在本机跑)
        build_demo()
        print("✅ 生产案例3 自检通过：GRPO 原理张量自检 + 三条【生产 GPU 训练】函数(Trainer/SFTTrainer/"
              "GRPOTrainer)已就绪 + Web。训练需 GPU 云，`accelerate launch` 启动。")
    elif len(sys.argv) > 2 and sys.argv[1] == "--train":
        {"ch2": train_classifier, "ch11": train_lora_sft, "ch12": train_grpo}[sys.argv[2]]()
    else:
        build_demo().queue().launch()
