"""
================================================================================
 分章项目 · Ch11 进阶 · 训练工程：Accelerate + 适配器分离加载 + QLoRA
================================================================================
 Ch11_LoRA指令微调.py 用手写循环训了 LoRA；本文件补上“上生产”要用的训练工程：
   ① Accelerate：一套代码透明支持 单卡/多卡/混合精度。核心就三个 API：
      accelerator.prepare(...) 包好模型/优化器/数据；accelerator.backward(loss) 反向。
   ② 适配器分离部署：训完只存【几 MB 的 LoRA 适配器】，推理时 基座 + PeftModel.from_pretrained
      重新拼起来 —— 一个基座可挂多个适配器(多租户/多任务共享显存)。
   ③ QLoRA：把基座 4-bit 量化再挂 LoRA，单卡也能微调大模型（需 bitsandbytes，Mac 跑不了，给参考）。
   完整对齐流程见 ../综合项目/项目3；生产多卡见 ../生产架构/生产03。
 跑：python3 Ch11_进阶_训练工程.py     # 真跑 Accelerate LoRA 训练 + 保存/重载适配器
================================================================================
"""
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch.optim import AdamW

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER_DIR = "/tmp/ch11-accel-adapter"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

DATA = [("What is the capital of France?", "The capital of France is Paris."),
        ("What is 2 plus 2?", "2 plus 2 equals 4."),
        ("Name a primary color.", "Red is a primary color."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo.")] * 4


def make_batches():
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
        tokenize=False) for q, a in DATA]
    enc = tok(texts, truncation=True, max_length=64, padding=True, return_tensors="pt")
    data = []
    for i in range(len(texts)):
        lab = enc["input_ids"][i].clone()
        lab[enc["attention_mask"][i] == 0] = -100
        data.append({"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": lab})
    return DataLoader(data, batch_size=4, shuffle=True)


# ==============================================================================
# ① Accelerate：prepare + backward(透明支持单卡/多卡/混合精度)
# ==============================================================================
def train_with_accelerate():
    # print("=" * 70, "\n① Accelerate 训练(accelerator.prepare / accelerator.backward)\n" + "=" * 70)
    from accelerate import Accelerator
    accelerator = Accelerator()                           # 自动选设备(mps/cuda/cpu)、管理混合精度/多卡
    print(f"  Accelerator 设备={accelerator.device}  混合精度={accelerator.mixed_precision}")

    model = get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL), LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    loader = make_batches()

    # ★一行把 模型/优化器/数据 都“准备”好(设备搬运、多卡包装、混合精度都在这里)
    model, opt, loader = accelerator.prepare(model, opt, loader)
    model.train()
    for _ in range(4):
        for batch in loader:
            loss = model(**batch).loss
            accelerator.backward(loss)                    # ★用它代替 loss.backward()(多卡/混合精度下正确)
            opt.step(); opt.zero_grad()
    print(f"  训练完成，末步 loss={loss.item():.3f}")
    # 存适配器：unwrap 拿回原模型再存(多卡下必须 unwrap)
    accelerator.unwrap_model(model).save_pretrained(ADAPTER_DIR)
    print(f"  LoRA 适配器已存到 {ADAPTER_DIR}（只有几 MB）")


# ==============================================================================
# ② 适配器分离部署：基座 + PeftModel.from_pretrained 重新拼
# ==============================================================================
def reload_adapter():
    # print("\n" + "=" * 70, "\n② 适配器分离加载(PeftModel.from_pretrained)\n" + "=" * 70)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(MODEL).to(dev)     # 干净基座
    model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(dev).eval()  # ★挂上刚训的适配器
    q = "What is the capital of France?"
    p = tok.apply_chat_template([{"role": "user", "content": q}],
                                tokenize=False, add_generation_prompt=True)
    enc = tok(p, return_tensors="pt").to(dev)
    out = model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id)
    print("  重载后推理:", tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()[:70])
    # print("  意义：基座只存一份，不同任务各存一个小适配器，推理时按需挂载(多租户/多任务省显存)。")
    # print("  也可 model.merge_and_unload() 把适配器并进基座权重，导出成独立模型部署。")


# ==============================================================================
# ③ QLoRA：基座 4-bit 量化 + LoRA —— 真实生产代码；需 CUDA GPU + bitsandbytes，本机不跑
# ==============================================================================
def qlora_train(train_dataset):
    """QLoRA = 4bit 量化基座 + LoRA，单张 24G 卡就能微调 7B/13B。
    真实生产写法(不是字符串)；本机跑不了(bitsandbytes 是 CUDA-only，Mac/mps 不支持)——
    在 GPU 机上传入 train_dataset 即可真训。"""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig   # 惰性导入
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,                       # ★基座权重 4-bit 量化(省 ~4 倍显存)
        bnb_4bit_quant_type="nf4",               # NormalFloat4(QLoRA 论文提出，比普通 int4 好)
        bnb_4bit_compute_dtype="bfloat16",       # 计算时反量化成 bf16
        bnb_4bit_use_double_quant=True,          # 双重量化再省一点
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)          # 量化模型训练前的必要处理(开梯度检查点等)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear",
        lora_dropout=0.05, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    # 之后训练/保存和普通 LoRA 完全一样(可直接交给 trl.SFTTrainer)：
    args = SFTConfig(output_dir="/mnt/ckpt/qlora", per_device_train_batch_size=4,
                     gradient_accumulation_steps=4, num_train_epochs=1, bf16=True,
                     gradient_checkpointing=True, logging_steps=10)
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_dataset)
    trainer.train()
    trainer.save_model("/mnt/ckpt/qlora-final")             # 只存 LoRA 适配器(几十 MB)
    return model


if __name__ == "__main__":
    train_with_accelerate()
    reload_adapter()
    # print("\n" + "=" * 70, "\n③ QLoRA(真实代码 qlora_train，需 GPU+bitsandbytes，本机不跑)\n" + "=" * 70)
    # print("  见本文件 qlora_train()：4bit 量化基座(BitsAndBytesConfig/nf4) + prepare_model_for_kbit_training")
    # print("  + LoRA + trl.SFTTrainer。GPU 机上 `qlora_train(你的数据集)` 即可真训。")
    print("✅ Ch11 进阶跑通：Accelerate(prepare/backward) → 适配器分离加载(PeftModel) → QLoRA(真实代码)。")
    # print("面试：Q Accelerate 解决什么?核心 API? Q 适配器为什么能分离部署? Q QLoRA 和 LoRA 区别?"
          # " (见 ../面试高频题库.py 五)")
