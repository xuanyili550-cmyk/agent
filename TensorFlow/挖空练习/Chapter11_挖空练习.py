"""
================================================================================
 Chapter 11 挖空练习 · LoRA 配置 + 包装模型（参数高效微调的核心）
================================================================================
 玩法：
   1) 把每个 ______ 换成正确代码（凭记忆，别翻笔记）
   2) 运行：python3 Chapter11_挖空练习.py   （下载 SmolLM2-135M ~270MB，秒级不训练）
   3) 没填的报 NameError；卡住 → 文件底部「答案区」
 目标：给一个真实小 LLM 套上 LoRA，只训练 <1% 的参数 —— 这是 LoRA 的立身之本。
       (完整训练闭环见 Chapter 11/Chapter11_LoRA微调实战.py)
================================================================================
"""
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model_name = "HuggingFaceTB/SmolLM2-135M"
base = AutoModelForCausalLM.from_pretrained(model_name)

# 练习1：LoRA 配置。填四个关键参数：
#   r=8（低秩矩阵的秩）、lora_alpha=16（缩放，一般=2*r）、
#   target_modules=["q_proj","v_proj"]（给注意力的 q/v 加 LoRA）、task_type="CAUSAL_LM"
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)

# 练习2：用配置把基座“包”成 LoRA 模型（peft 的哪个函数？参数：base, lora_config）
lora_model = get_peft_model(base, lora_config)

# 练习3：打印可训练参数占比（peft 模型自带的方法名？）
lora_model.print_trainable_parameters()

# 自检：可训练参数应远小于 1%（LoRA 冻结基座，只训低秩适配器）
trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in lora_model.parameters())
pct = trainable / total * 100
print(f"\n可训练 {trainable:,} / 全部 {total:,} = {pct:.2f}%")
assert pct < 1.0, "LoRA 可训练占比应 <1%，检查 r / target_modules 填对没"
print("自检通过 ✅：只训不到 1% 的参数就能微调整个模型，这就是 LoRA 省显存的关键。")


# ==============================================================================
#  答案区（卡住再看！）
# ------------------------------------------------------------------------------
#  1: LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
#                target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM", bias="none")
#  2: get_peft_model(base, lora_config)
#  3: lora_model.print_trainable_parameters()
# ==============================================================================
