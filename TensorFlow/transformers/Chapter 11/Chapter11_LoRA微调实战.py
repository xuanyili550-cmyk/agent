"""
================================================================================
 Chapter 11 · LoRA 微调实战（真实数据 + 真实 LLM + PEFT/TRL 完整闭环）
================================================================================
 “能写进简历/能上线”的 LoRA 参数高效微调（PEFT），不是玩具：
   · 真实指令数据集 databricks/databricks-dolly-15k、真实小 LLM SmolLM2-135M
   · peft LoraConfig + trl SFTTrainer：只训 <0.4% 参数就能微调整个模型
   · 完整闭环：配置 → 训练 → 打印可训练参数占比 → 保存适配器 → 加载推理 →
     合并适配器回基座(merge_and_unload) → 部署形态说明
   · 全程能在 Mac(mps/cpu)真跑，几十秒训完(小步数)

 为什么 LoRA 是当下最抢手的技能：全量微调一个大模型要存全部权重的梯度+优化器状态，
 消费级显卡根本放不下。LoRA 冻结原权重，只在每层旁边插一对低秩小矩阵(A×B)去学“增量”，
 可训练参数骤降到 <1%，显存/存储大幅下降，还能一个基座挂多个适配器(多任务切换)。

 用法：
   python3 Chapter11_LoRA微调实战.py                 # 真训练(小步数，~1 分钟)
   python3 Chapter11_LoRA微调实战.py --steps 100 --n 800   # 训久一点、数据更多

 ⚠ QLoRA(4-bit)需要 bitsandbytes，它是 CUDA-only，Mac 跑不了 —— 本文件用普通 LoRA；
   QLoRA 的写法在文件末尾以“云端 GPU 参考”给出。
================================================================================
"""

import argparse
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    model_name: str = "HuggingFaceTB/SmolLM2-135M"        # 真实小 LLM(1.35 亿参数)
    dataset_name: str = "databricks/databricks-dolly-15k"  # 真实指令微调数据集
    n_train: int = 300                 # 训练子集(快速验证；调大更好)
    max_steps: int = 40                # 训练步数(Mac 上先小步跑通)
    batch_size: int = 2
    grad_accum: int = 4                # 有效 batch = 2*4 = 8
    lr: float = 2e-4                   # LoRA 通常比全量微调用更大的 lr
    max_length: int = 256
    # --- LoRA 超参 ---
    lora_r: int = 8                    # 低秩矩阵的秩：越大容量越大、可训参数越多(常用 8/16/32)
    lora_alpha: int = 16               # 缩放系数：一般设成 2*r
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])  # 给哪些层加 LoRA
    output_dir: str = "smollm2-dolly-lora"


PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


def build_dataset(cfg):
    from datasets import load_dataset
    ds = load_dataset(cfg.dataset_name, split="train").shuffle(seed=42)
    ds = ds.select(range(min(cfg.n_train, len(ds))))
    # 把结构化字段拼成“指令-回复”训练文本(SFT 的标准做法)
    def to_text(ex):
        return {"text": PROMPT.format(instruction=ex["instruction"],
                                      response=ex["response"])}
    return ds.map(to_text, remove_columns=ds.column_names)


def train(cfg: Config):
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 设备={device}  基座={cfg.model_name}  数据集={cfg.dataset_name}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset(cfg)
    print(f">>> 训练样本 {len(train_ds)} 条，示例：\n{train_ds[0]['text'][:120]}...")

    # LoRA 配置：冻结基座，只在 q_proj/v_proj 旁插低秩矩阵
    lora_config = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules, task_type="CAUSAL_LM", bias="none")

    args = SFTConfig(
        output_dir=cfg.output_dir, max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr, warmup_steps=max(1, cfg.max_steps // 10),
        logging_steps=10, max_length=cfg.max_length,
        dataset_text_field="text", report_to=[],
    )
    # 传 peft_config 给 SFTTrainer，它会自动把模型包成 LoRA(这是 TRL 的现代写法)
    trainer = SFTTrainer(model=cfg.model_name, args=args,
                         train_dataset=train_ds, peft_config=lora_config)

    print("\n>>> LoRA 可训练参数占比(核心卖点)：")
    trainer.model.print_trainable_parameters()   # 打印 trainable% —— 应 <1%

    trainer.train()
    trainer.save_model(cfg.output_dir)            # 只保存适配器(几 MB，不是整个模型)
    print(f">>> 训练完成，LoRA 适配器已存到 {cfg.output_dir}/")
    adapter_mb = sum(os.path.getsize(os.path.join(cfg.output_dir, f))
                     for f in os.listdir(cfg.output_dir)
                     if f.endswith((".safetensors", ".bin"))) / 1e6
    print(f">>> 适配器大小仅 {adapter_mb:.1f} MB(对比基座 ~270MB —— 这就是 LoRA 省存储的体现)")


def inference_and_merge(cfg: Config):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    if not os.path.isdir(cfg.output_dir):
        print("(还没训练出适配器，跳过)"); return

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 部署形态一：基座 + 适配器分开加载(一个基座可挂多个适配器切任务) ---
    base = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    lora_model = PeftModel.from_pretrained(base, cfg.output_dir).eval()
    prompt = PROMPT.format(instruction="Name three colors.", response="")
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = lora_model.generate(**inputs, max_new_tokens=30, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
    print("\n>>> 基座+适配器 推理：")
    print("   ", tokenizer.decode(out[0], skip_special_tokens=True).replace(prompt, "").strip()[:120])

    # --- 部署形态二：把适配器合并回基座，得到一个独立模型(推理零额外开销) ---
    merged = lora_model.merge_and_unload()        # 合并 LoRA 增量到原权重
    print(">>> merge_and_unload 完成：适配器已并入基座，可当普通模型保存/部署(推理无 LoRA 开销)")
    # merged.save_pretrained("smollm2-dolly-merged")   # 生产上常存这个合并版直接上线


# ==============================================================================
# QLoRA(4-bit)—— 需要 CUDA + bitsandbytes，Mac 跑不了，这里给云端参考写法
# ==============================================================================
# from transformers import BitsAndBytesConfig
# import torch
# bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
#                          bnb_4bit_compute_dtype=torch.bfloat16,
#                          bnb_4bit_use_double_quant=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb,
#                                              device_map="auto")
# from peft import prepare_model_for_kbit_training
# model = prepare_model_for_kbit_training(model)
# # 之后同样 LoraConfig + SFTTrainer(peft_config=...) —— 这就是 QLoRA：
# # 基座用 4-bit 量化加载(省一半以上显存) + LoRA 只训低秩适配器，
# # 让 7B/13B 模型能在单张消费级 24G 卡上微调。


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--n", type=int, default=300)
    args = p.parse_args()
    cfg = Config(max_steps=args.steps, n_train=args.n)
    train(cfg)
    inference_and_merge(cfg)
