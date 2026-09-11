"""
================================================================================
 案例3 · LoRA 指令微调（整合 Ch11 + Ch7）—— 完整可跑，照着手写练熟
================================================================================
 目标：用 LoRA 高效微调一个真实小 LLM，让它学会“按指令作答”。
 整合了哪几章、每步为什么用这个技术：
   [Ch11 LoRA]   为什么用 LoRA——全量微调要存全部权重的梯度+优化器状态，消费级显卡放不下；
                 LoRA 冻结原权重、只在每层旁插一对低秩小矩阵，可训练参数骤降到 <1%，还能存成
                 几 MB 的适配器、一个基座挂多个适配器切任务。
   [Ch7 训练工程] 为什么要这些——学习率 warmup(开头别太猛)、梯度累积(小 batch 模拟大 batch 省显存)、
                 梯度裁剪(防梯度爆炸)、Accelerate(设备/混合精度透明)。把模型“真正训起来”的工程。
   [Ch11 SFT]    指令数据格式：把“指令→回复”拼成一段文本让模型模仿(监督微调)。
   [Ch11 部署]   训完保存适配器 → 加载基座+适配器推理 → merge_and_unload 合并回基座。

 直接运行：python3 案例3_LoRA指令微调.py     # 真训练(小步数，~1 分钟)，看它学会指令格式
================================================================================
"""
import os
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
ADAPTER = "案例3-lora-adapter"


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def lora_param_scaling(cfg):
    """[知识点·r/alpha/target_modules 对参数量的影响] 直接从模型结构算 LoRA 可训练参数，
    看清三个旋钮怎么影响“训多少参数”——面试常问、也决定你显存够不够。
    LoRA 给某个 in×out 的权重旁插 A(in×r)+B(r×out)，新增 = r*(in+out) 个参数。
      · r 翻倍 → 适配器参数翻倍(容量↑但省得↓)；alpha 只是缩放，不改参数量(常设 2r)。
      · target_modules 加得越多(q/k/v/o + mlp)→ 覆盖层越多、参数越多、效果通常越好。"""
    h = cfg.hidden_size
    L = cfg.num_hidden_layers
    kv = getattr(cfg, "head_dim", h // cfg.num_attention_heads) * \
        getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)      # GQA 下 k/v 维度更小
    inter = getattr(cfg, "intermediate_size", 4 * h)
    # 每个投影层的 (输入维, 输出维)
    dims = {"q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h),
            "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h)}
    # [自检] LoRA 参数量随 r/target_modules 变化 教学演示表头（噪音，已静音）
    for tms in (["q_proj", "v_proj"], ["q_proj", "k_proj", "v_proj", "o_proj"], list(dims)):
        for r in (8, 16):
            per_layer = sum(r * (i + o) for m, (i, o) in dims.items() if m in tms)
            total = per_layer * L
            # [自检] 逐组打印 target/r → 可训练参数量（噪音，已静音）
    # [自检] 结论：r 翻倍→参数翻倍；q/v 扩到 all-linear→参数涨数倍；alpha 不影响参数量（已静音）


def load_dolly(n=200):
    # 真实指令数据集 dolly-15k(JSONL)。pandas 直读 hf://，绕开本机坏掉的 datasets.load_dataset。
    df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl",
                      lines=True).sample(n=n, random_state=42)
    return [PROMPT.format(instruction=r["instruction"], response=r["response"])
            for _, r in df.iterrows()]


def main():
    device = pick_device()
    print(f">>> 设备={device}  基座={MODEL}  数据=dolly-15k")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- [Ch11] 给基座套 LoRA：冻结原权重，只训低秩适配器 ----
    base = AutoModelForCausalLM.from_pretrained(MODEL)
    # r=秩(容量,常8/16/32); lora_alpha=缩放(常设 2r，实际缩放=alpha/r); dropout 防过拟合;
    # target_modules=给哪些层加(注意力 q/v 性价比高，要更强就 all-linear); bias="none" 不训偏置。
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", bias="none")).to(device)
    model.print_trainable_parameters()        # ← 应显示可训练 <1%
    lora_param_scaling(base.config)           # [知识点] 看 r/target_modules 怎么影响参数量

    # ---- 数据：指令文本 → token（因果 LM 的标签就是 input_ids 本身，模型内部会错一位对齐）----
    texts = load_dolly(200)
    enc = tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
    dataset = []
    for i in range(len(texts)):
        labels = enc["input_ids"][i].clone()
        # ★把 [PAD] 位置标签设 -100：PyTorch 交叉熵默认 ignore_index=-100，即这些位置不算 loss。
        # 为什么必须这样：填充只是为了把 batch 补成矩形，本身没有语义；若在 [PAD] 上算 loss，
        # 模型会去“学着预测填充”，白白污染梯度。同理，只想让模型学“回复”不学“指令”时，
        # 也可把指令部分的 label 设 -100(prompt masking)，只在 Response 段回传 loss。
        labels[enc["attention_mask"][i] == 0] = -100
        dataset.append({"input_ids": enc["input_ids"][i],
                        "attention_mask": enc["attention_mask"][i], "labels": labels})
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # ---- [Ch7] 训练工程：AdamW + warmup 调度 + 梯度累积 + 梯度裁剪 ----
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    grad_accum, steps = 2, 20
    sched = get_scheduler("linear", optimizer, num_warmup_steps=2, num_training_steps=steps)

    model.train()
    step = 0
    data_iter = iter(loader)
    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); sched.step(); optimizer.zero_grad()
        if step % 5 == 0:
            print(f"   step {step}/{steps} loss={loss.item()*grad_accum:.3f}")
        step += 1

    # ---- [Ch11] 保存适配器(几 MB) + 推理验证学没学会指令格式 ----
    model.save_pretrained(ADAPTER)
    mb = sum(os.path.getsize(os.path.join(ADAPTER, f)) for f in os.listdir(ADAPTER)
             if f.endswith((".safetensors", ".bin"))) / 1e6
    print(f"\n>>> 适配器已存 {ADAPTER}/ (仅 {mb:.1f} MB，对比基座 ~270MB)")

    model.eval()
    for q in ["Name three colors.", "What is the capital of Italy?"]:
        p = PROMPT.format(instruction=q, response="")
        out = model.generate(**tok(p, return_tensors="pt").to(device), max_new_tokens=25,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        ans = tok.decode(out[0], skip_special_tokens=True).replace(p, "").strip()
        print(f"   Q: {q}\n   A: {ans[:80]}")

    # [Ch11] 合并适配器回基座(部署时推理零额外开销)
    # merge_and_unload: 把 B@A 增量加回原权重 W→W+BA，得到普通模型，推理无 LoRA 分支开销;
    # 代价是失去“换适配器切任务”的灵活。要多任务/热切换就别 merge，保留基座+多个适配器。
    merged = model.merge_and_unload()
    print(f"\n✅ LoRA 微调跑通：只训 <1% 参数(Ch11) + 完整训练工程(Ch7)；merge 后 = {type(merged).__name__}")


# ==============================================================================
# 更多知识点
# ==============================================================================
# · [梯度累积] 显存放不下大 batch 时，累积 N 个小 batch 的梯度再 step，等效 batch×N(loss 要÷N)。
#   本例 grad_accum=2。它省显存但不省时间(照样前反向 N 次)。
# · [梯度裁剪] clip_grad_norm_(…,1.0)：把梯度整体范数截到阈值内，防某步梯度爆炸把训练带崩(loss→NaN)。
# · [QLoRA] = 基座用 4-bit(NF4)量化加载(省一半以上显存) + 只训 LoRA 适配器。让 7B/13B 在单张
#   消费级卡上微调;精度损失很小。需 bitsandbytes(CUDA);Mac/mps 上跑不了，本例才用 fp32 小模型。
# · [显存阶梯] 放不下就依次上:梯度累积→梯度检查点(重算激活换显存)→混合精度→LoRA→QLoRA→ZeRO/offload。

# ==============================================================================
# 面试题(LoRA/PEFT)
# ==============================================================================
# Q1: LoRA 为什么能把可训练参数降到 <1%？r/alpha/target_modules 各起什么作用？
# A : 冻结原权重 W，只在旁边训一对低秩矩阵 A(in×r)、B(r×out)，用 W+BA 近似更新;r 很小故参数极少。
#     r=容量(越大越强越费)、alpha=缩放(实际缩放 alpha/r，不改参数量)、target_modules=加到哪些层
#     (q/v 性价比高，all-linear 效果更好但更费)。
# Q2: 为什么因果 LM 训练要把 [PAD] 的 label 设 -100？
# A : 交叉熵 ignore_index=-100 会跳过这些位置。填充无语义，在上面算 loss 等于让模型学预测填充，
#     污染梯度。同理可用 prompt masking 只对 Response 段回传 loss，让模型专注学“怎么答”。
# Q3: QLoRA 和 LoRA 区别？什么时候用？
# A : QLoRA 额外把基座 4-bit(NF4)量化加载，显存再省一半以上，让大模型能在单卡消费级微调;精度损失小。
#     显存紧张跑大模型时用(需 CUDA+bitsandbytes)。LoRA 本身不量化基座。
# Q4: merge_and_unload 干嘛？合并前后怎么取舍？
# A : 把 LoRA 增量并回基座得普通模型，推理零额外开销、可直接部署/导出;但失去换适配器的灵活。
#     单任务上线就 merge;多任务热切换就保留基座+多适配器不 merge。
# Q5: 什么时候该 LoRA 微调，什么时候用 RAG/提示词？
# A : 要“新知识/私有知识”→RAG(知识会变，微调追不上);要“新能力/固定格式/领域风格”→SFT+LoRA。
#     常组合:微调管“怎么答”+RAG 管“答什么”。微调是较贵的手段，先试提示+RAG。


if __name__ == "__main__":
    main()
