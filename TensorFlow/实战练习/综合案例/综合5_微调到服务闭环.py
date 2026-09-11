"""
================================================================================
 综合案例5 · 微调到服务闭环（整合 Ch11 LoRA/SFT + 聊天模板 + 保存/重载 + 推理服务）
================================================================================
 端到端走一遍“把基座模型变成能用的服务”：
   数据(聊天模板) → LoRA 微调(只训<1%参数，Mac 真跑) → 保存适配器 → 基座+适配器重载 → 推理(服务)
 整合：Ch11 apply_chat_template + LoRA + PeftModel 分离部署；训练/推理分离(生产铁律)。
 跑：python3 综合5_微调到服务闭环.py     # 真训练小步数 ~1-2 分钟
================================================================================
"""
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch.optim import AdamW

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER = "/tmp/综合5-adapter"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 指令数据(用聊天模板；这里用固定小样本让 Mac 秒级可跑)
PAIRS = [("What is the capital of France?", "The capital of France is Paris."),
         ("What is 2 plus 2?", "2 plus 2 equals 4."),
         ("Name a primary color.", "Red is a primary color."),
         ("What is the capital of Japan?", "The capital of Japan is Tokyo.")] * 5


def make_loader():
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
        tokenize=False) for q, a in PAIRS]
    enc = tok(texts, truncation=True, max_length=64, padding=True, return_tensors="pt")
    data = []
    for i in range(len(texts)):
        lab = enc["input_ids"][i].clone()
        lab[enc["attention_mask"][i] == 0] = -100        # 不在 PAD 上算 loss
        data.append({"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": lab})
    return DataLoader(data, batch_size=4, shuffle=True)


def train():
    """① LoRA 微调(冻结基座，只训低秩适配器)。"""
    # print(">>> ① LoRA 微调(Ch11)...")
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL), LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).to(DEV)
    model.print_trainable_parameters()
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    loader = make_loader()
    model.train()
    it, step = iter(loader), 0
    while step < 30:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        b = {k: v.to(DEV) for k, v in b.items()}
        model(**b).loss.backward(); opt.step(); opt.zero_grad(); step += 1
    model.save_pretrained(ADAPTER)                        # ② 保存适配器(几 MB)
    print(f">>> ② 适配器已保存到 {ADAPTER}")


def serve(question):
    """③ 服务：基座 + 适配器重载(训练/推理分离)，推理。"""
    base = AutoModelForCausalLM.from_pretrained(MODEL).to(DEV)
    model = PeftModel.from_pretrained(base, ADAPTER).to(DEV).eval()
    p = tok.apply_chat_template([{"role": "user", "content": question}],
                                tokenize=False, add_generation_prompt=True)
    enc = tok(p, return_tensors="pt").to(DEV)
    out = model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    print(f">>> 设备={DEV}  基座={MODEL}")
    train()
    # print(">>> ③ 服务(基座+适配器重载 → 推理)：")
    for q in ["What is the capital of France?", "What is 2 plus 2?"]:
        ans = serve(q)
        print(f"   Q: {q}\n   A: {ans[:60]}")
    assert "Paris" in serve("What is the capital of France?")
    print("\n✅ 综合5 跑通：聊天模板→LoRA微调→保存适配器→基座+适配器重载→推理服务（整合 Ch11+部署）。")
    # print("   生产：多卡训练(../chapter实战/生产03) + 推模型仓库 + vLLM 服务(生产01/05)。")
