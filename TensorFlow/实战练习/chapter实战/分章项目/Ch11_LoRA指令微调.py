"""
================================================================================
 分章项目 · Ch11 · LoRA 指令微调（贴 HF Ch11：聊天模板 + SFT + LoRA）
================================================================================
 HF 课程 Ch11“微调 LLM”的三件核心，用可运行代码复现(在真实指令数据 dolly 上微调 SmolLM2)：
   ① 聊天模板：apply_chat_template 把 user/assistant 消息拼成模型认的格式(ChatML)，别手拼特殊符。
   ② LoRA：冻结基座，只在每层旁插低秩小矩阵，可训练参数 <1%，适配器只几 MB。
   ③ SFT 监督微调：用“指令→回复”教基座听指令；训完看它学会指令格式。
   生产用 trl.SFTTrainer 一键搞定(见文末)；这里手写训练循环是为看清 SFT 内部。
   完整对齐流程(SFT+GRPO)见 ../综合项目/项目3；生产多卡见 ../生产架构/生产03。
 跑：python3 Ch11_LoRA指令微调.py     # 真训练小步数，~1-2 分钟
================================================================================
"""
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch.optim import AdamW

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# ==============================================================================
# ① 聊天模板：apply_chat_template 把消息拼成模型认的格式
# ==============================================================================
def show_chat_template():
    # print("=" * 70, "\n① 聊天模板 apply_chat_template(别手拼特殊符)\n" + "=" * 70)
    messages = [{"role": "user", "content": "What is the capital of Italy?"},
                {"role": "assistant", "content": "The capital of Italy is Rome."}]
    text = tok.apply_chat_template(messages, tokenize=False)
    print("  messages →\n" + "\n".join("    " + l for l in text.splitlines()))
    # print("  推理时用 add_generation_prompt=True，让模型接着 assistant 位置往下生成。")


# ==============================================================================
# ② + ③ LoRA + SFT：用聊天模板格式的指令数据微调
# ==============================================================================
def sft_lora(steps=20):
    # print("\n" + "=" * 70, "\n②③ LoRA + SFT 监督微调\n" + "=" * 70)
    df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl",
                      lines=True)
    df = df[df["context"].str.len() == 0].sample(n=150, random_state=42)   # 取纯指令题
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": r["instruction"]},
         {"role": "assistant", "content": r["response"]}], tokenize=False) for _, r in df.iterrows()]
    enc = tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
    data = []
    for k in range(len(texts)):
        lab = enc["input_ids"][k].clone()
        lab[enc["attention_mask"][k] == 0] = -100      # 不在 PAD 上算 loss
        data.append({"input_ids": enc["input_ids"][k], "attention_mask": enc["attention_mask"][k], "labels": lab})

    # ② LoRA：冻结基座，只训低秩适配器
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL), LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM")).to(DEV)
    model.print_trainable_parameters()                 # ← <1%

    # ③ SFT：手写训练循环(trl.SFTTrainer 内部也是这些)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    loader = DataLoader(data, batch_size=4, shuffle=True)
    model.train()
    it, step = iter(loader), 0
    while step < steps:
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        batch = {k: v.to(DEV) for k, v in batch.items()}
        model(**batch).loss.backward(); opt.step(); opt.zero_grad(); step += 1
    return model


def generate(model, question):
    p = tok.apply_chat_template([{"role": "user", "content": question}],
                                tokenize=False, add_generation_prompt=True)
    enc = tok(p, return_tensors="pt").to(DEV)
    out = model.generate(**enc, max_new_tokens=30, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    print(f">>> 设备={DEV}\n")
    show_chat_template()
    model = sft_lora(steps=20)
    model.eval()
    print("\n  推理(LoRA 微调后)：", generate(model, "What is the capital of Italy?")[:80])
    print("\n✅ Ch11 跑通：聊天模板 → LoRA(只训<1%) → SFT 训练 → 推理。")
    # print("  生产捷径：trl.SFTTrainer(model, args=SFTConfig(...), train_dataset=ds, peft_config=lora)")
    # print("面试：Q LoRA 原理?为什么省? Q r/alpha/target_modules 怎么设? Q 为什么用聊天模板不手拼?"
          # " (见 ../面试高频题库.py 五)")
