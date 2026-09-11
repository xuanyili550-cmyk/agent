"""
================================================================================
 Chapter 11 案例闯关 · LoRA / SFT 生产实战（5 关，真实模型真训练）
================================================================================
 全部生产写法:真实 LLM(SmolLM2-135M) + 真实数据(dolly-15k) + peft/trl,能在 Mac 真跑。

 用法：
   python3 Chapter11_案例闯关_LoRA_SFT实战.py            # 看菜单
   python3 Chapter11_案例闯关_LoRA_SFT实战.py 1          # 只跑第 1 关
   python3 Chapter11_案例闯关_LoRA_SFT实战.py all         # 全部

 关卡：
   1  LoRA 配置详解     r/alpha/target_modules 对“可训练参数量”的真实影响(不训练,秒出)
   2  真 LoRA 微调       SmolLM2 on dolly 训几十步,存适配器(~30-60s)
   3  加载适配器推理     PeftModel 加载 base+adapter,生成回复
   4  合并适配器        merge_and_unload 把 LoRA 并回基座(部署无额外开销)
   5  聊天模板实战       apply_chat_template 多模型对比 + generation prompt

 说明:2/3/4 关有依赖(先训才能推理/合并);单独跑 3/4 会自动补训一个小适配器。
================================================================================
"""

import os
import sys

MODEL = "HuggingFaceTB/SmolLM2-135M"
ADAPTER_DIR = "smollm2-lora-case"
PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


def title(n, text):
    print("\n" + "=" * 72 + f"\n  第 {n} 关：{text}\n" + "=" * 72)


def _small_dataset(n=120):
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=42)
    ds = ds.select(range(n))
    return ds.map(lambda e: {"text": PROMPT.format(instruction=e["instruction"],
                                                   response=e["response"])},
                  remove_columns=ds.column_names)


def _train_adapter(steps=30, out=ADAPTER_DIR):
    """真 LoRA 微调并保存适配器(2/3/4 关共用)。"""
    from transformers import AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", bias="none")
    args = SFTConfig(output_dir=out, max_steps=steps, per_device_train_batch_size=2,
                     gradient_accumulation_steps=4, learning_rate=2e-4,
                     warmup_steps=max(1, steps // 10), logging_steps=10,
                     max_length=256, dataset_text_field="text", report_to=[])
    trainer = SFTTrainer(model=MODEL, args=args, train_dataset=_small_dataset(),
                         peft_config=lora)
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(out)
    return out


def _ensure_adapter():
    if not os.path.isdir(ADAPTER_DIR):
        print("  (没有现成适配器,先快速训一个 15 步...)")
        _train_adapter(steps=15)


# ==============================================================================
# 第 1 关：LoRA 配置详解(参数量的真实影响)
# ==============================================================================
def case_1():
    title(1, "LoRA 配置：r / target_modules 对可训练参数量的影响")
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    def count(r, target):
        base = AutoModelForCausalLM.from_pretrained(MODEL)
        m = get_peft_model(base, LoraConfig(r=r, lora_alpha=2 * r,
                                            target_modules=target, task_type="CAUSAL_LM"))
        tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in m.parameters())
        return tr, tot

    print("  基座 SmolLM2-135M，对比不同 LoRA 配置的“可训练参数量”：\n")
    for r, target, note in [(4, ["q_proj", "v_proj"], "小秩+仅注意力"),
                            (16, ["q_proj", "v_proj"], "大秩+仅注意力"),
                            (8, "all-linear", "所有线性层")]:
        tr, tot = count(r, target)
        print(f"   r={r:<2} target={str(target):24} {note:12} "
              f"可训练={tr:>9,} ({tr/tot*100:.2f}%)")
    print("\n  要点：r 越大 / target_modules 越多 → 可训练参数越多、容量越强，但省的越少。")
    print("        常用起点：r=8/16、target=注意力的 q_proj/v_proj(性价比高)。")


# ==============================================================================
# 第 2 关：真 LoRA 微调
# ==============================================================================
def case_2():
    title(2, "真 LoRA 微调 SmolLM2 on dolly(存适配器)")
    _train_adapter(steps=30)
    mb = sum(os.path.getsize(os.path.join(ADAPTER_DIR, f)) for f in os.listdir(ADAPTER_DIR)
             if f.endswith((".safetensors", ".bin"))) / 1e6
    print(f"  ✅ 训练完成，适配器存到 {ADAPTER_DIR}/ (仅 {mb:.1f} MB)")
    print("  要点：只训 <1% 参数，存的是小小的适配器,不是整个模型。")


# ==============================================================================
# 第 3 关：加载适配器推理
# ==============================================================================
def case_3():
    title(3, "加载 base + 适配器做推理")
    _ensure_adapter()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
    for q in ["Name three colors.", "What is the capital of Japan?"]:
        p = PROMPT.format(instruction=q, response="")
        out = model.generate(**tok(p, return_tensors="pt"), max_new_tokens=25,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        ans = tok.decode(out[0], skip_special_tokens=True).replace(p, "").strip()
        print(f"  Q: {q}\n  A: {ans[:100]}\n")
    print("  要点：部署时基座 + 适配器分开加载,一个基座可挂多个适配器切任务。")


# ==============================================================================
# 第 4 关：合并适配器回基座
# ==============================================================================
def case_4():
    title(4, "merge_and_unload：把 LoRA 并回基座")
    _ensure_adapter()
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(MODEL)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    before = type(model).__name__
    merged = model.merge_and_unload()
    after = type(merged).__name__
    print(f"  合并前类型: {before}(带 LoRA 包装)")
    print(f"  合并后类型: {after}(普通模型,LoRA 增量已并入权重)")
    print("  要点：合并后推理零额外开销,可当普通模型 save_pretrained 直接上线;")
    print("        但就失去了“换适配器”的灵活性 —— 按部署需求二选一。")


# ==============================================================================
# 第 5 关：聊天模板实战
# ==============================================================================
def case_5():
    title(5, "apply_chat_template 多模型对比 + generation prompt")
    from transformers import AutoTokenizer
    msgs = [{"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"}]
    for name in ["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]:
        try:
            t = AutoTokenizer.from_pretrained(name)
            print(f"  【{name.split('/')[-1]}】", repr(t.apply_chat_template(msgs, tokenize=False)[:110]))
        except Exception as e:
            print(f"  (跳过 {name}: {type(e).__name__})")
    t = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    withgen = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print("\n  add_generation_prompt=True 末尾(推理时用):", repr(withgen[-35:]))
    enc = t.apply_chat_template(msgs, tokenize=True)
    print(f"  tokenize=True → {len(enc['input_ids'])} 个 input_ids(可直接喂模型)")
    print("  要点：永远用 apply_chat_template,别手拼特殊标记;推理加 add_generation_prompt。")


CASES = {i: globals()[f"case_{i}"] for i in range(1, 6)}


def run(choice):
    if choice == "all":
        for i in sorted(CASES):
            CASES[i]()
    elif choice.isdigit() and int(choice) in CASES:
        CASES[int(choice)]()
    else:
        print(f"没有第 {choice} 关,可选 {sorted(CASES)} 或 all")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
