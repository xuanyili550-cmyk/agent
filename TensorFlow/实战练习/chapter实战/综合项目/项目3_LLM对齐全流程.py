"""
================================================================================
 综合项目3 · LLM 对齐全流程（整合 Ch3 + Ch7 + Ch11 + Ch12）
================================================================================
 把一个基座 LLM 对齐成“会听指令、能推理”的助手，走完整三段：
   数据(聊天模板) → SFT+LoRA 微调 → GRPO 强化 → 评估 → 服务(推理)
 整合哪些章、为什么用：
   [Ch3 预训练模型] 从 Hub 拉一个基座小 LLM(SmolLM2) 当起点。
   [Ch11 聊天模板]  apply_chat_template 把对话拼成模型认的格式；SFT 用“指令→回复”教它听话。
   [Ch11 LoRA]     只训 <1% 参数微调，消费级也扛得住；参考策略靠“关掉 LoRA 的基座”。
   [Ch7 训练工程]  优化器/warmup 调度/梯度累积/裁剪——把模型真正训起来。
   [Ch12 GRPO]     用“可验证奖励(答对才给分)”进一步强化推理(见本文件 GRPO 段/../案例4)。
   为什么这个顺序：预训练(会续写)→SFT(会听指令)→RL(对齐偏好/强化推理)，层层递进。

 本地跑：python3 项目3_LLM对齐全流程.py           # 真跑 SFT+LoRA(小步数~1分钟) + 评估 + 推理
        python3 项目3_LLM对齐全流程.py grpo       # 额外跑一小段 GRPO(慢)
 生产版(多卡 SFT / trl.GRPOTrainer+vLLM)见 ../生产架构/生产03、生产04。
================================================================================
"""
import os
import sys
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER = "项目3-sft-lora"


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def load_sft_data(tok, n=150):
    """[Ch3+Ch11] 真实指令数据(dolly) → 用聊天模板拼成训练文本。"""
    df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl",
                      lines=True)
    df = df[df["context"].str.len() == 0].sample(n=n, random_state=42)   # 取无上下文的纯指令题
    texts = []
    for _, r in df.iterrows():
        # apply_chat_template：把 user/assistant 拼成模型认识的 ChatML 格式(Ch11)
        texts.append(tok.apply_chat_template(
            [{"role": "user", "content": r["instruction"]},
             {"role": "assistant", "content": r["response"]}], tokenize=False))
    return texts


def sft_train(steps=25):
    dev = pick_device()
    print(f">>> [SFT] 设备={dev}  基座={MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL)
    # [Ch11] LoRA
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM", lora_dropout=0.05)).to(dev)
    model.print_trainable_parameters()

    texts = load_sft_data(tok, 150)
    enc = tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
    data = []
    for i in range(len(texts)):
        labels = enc["input_ids"][i].clone()
        labels[enc["attention_mask"][i] == 0] = -100        # 不在 [PAD] 上算 loss
        data.append({"input_ids": enc["input_ids"][i],
                     "attention_mask": enc["attention_mask"][i], "labels": labels})
    loader = DataLoader(data, batch_size=4, shuffle=True)

    # [Ch7] 训练工程：AdamW + warmup + 梯度累积 + 裁剪
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    accum = 2
    sched = get_scheduler("linear", opt, num_warmup_steps=2, num_training_steps=steps)
    model.train()
    it, step = iter(loader), 0
    while step < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {k: v.to(dev) for k, v in batch.items()}
        (model(**batch).loss / accum).backward()
        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        step += 1
    model.save_pretrained(ADAPTER)
    print(f">>> [SFT] 完成，适配器存到 {ADAPTER}/")
    return tok, model, dev


def evaluate(tok, model, dev):
    """[评估] 看模型学没学会指令格式(定性)。生产用基准测试集+人工评。"""
    # print("\n>>> [评估] 指令跟随测试：")
    model.eval()
    for q in ["Name three primary colors.", "What is the capital of Japan?"]:
        p = tok.apply_chat_template([{"role": "user", "content": q}],
                                    tokenize=False, add_generation_prompt=True)
        out = model.generate(**tok(p, return_tensors="pt").to(dev), max_new_tokens=30,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        ans = tok.decode(out[0][tok(p, return_tensors="pt")["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        print(f"   Q: {q}\n   A: {ans[:80]}")


def grpo_stage():
    """[Ch12] GRPO 强化(可验证奖励)。这里只提示，完整可跑手写版见 ../案例4；生产见 ../生产架构/生产04。"""
    # print("\n>>> [GRPO] 强化阶段：用‘答对才给分’的可验证奖励 + 组内优势进一步强化。")
    # print("    完整手写可跑版：python3 ../案例4_GRPO强化学习.py")
    # print("    生产版(trl.GRPOTrainer+vLLM)：../生产架构/生产04_GRPO_trl_vLLM.py")


def main():
    tok, model, dev = sft_train(steps=25)
    evaluate(tok, model, dev)
    grpo_stage()
    print("\n✅ 对齐全流程跑通：聊天模板(Ch11)→SFT+LoRA(Ch11/7)→评估→GRPO强化(Ch12)。")
    # print("   生产：多卡 SFT(生产03) + trl.GRPOTrainer(生产04) + 推模型仓库 + vLLM 服务(生产01)。")


# ==============================================================================
# 面试题
# ==============================================================================
# Q: 预训练 / SFT / RLHF(GRPO) 三段各解决什么？(见面试题库 五.LoRA、六.GRPO)
# Q: 为什么 SFT 用 LoRA 而不全量微调？(见面试题库 五)
# Q: GRPO 和 PPO 区别？优势怎么算？(见面试题库 六)
# Q: SFT 时为什么要在 [PAD] 上把标签设 -100？
# A: 不在填充 token 上计算损失，否则模型会学“预测 padding”，loss 被污染、训练变差。
# Q: 聊天模板(apply_chat_template)为什么不能手拼特殊标记？
# A: 各模型模板不同(ChatML/Llama/Mistral)，手拼极易错；分词器自带模板，推理还要加 add_generation_prompt。

if __name__ == "__main__":
    main()
    if len(sys.argv) > 1 and sys.argv[1] == "grpo":
        # print("\n跑 GRPO 小段：")
        os.system(f"{sys.executable} ../案例4_GRPO强化学习.py --steps 6 --group 3")
