"""
================================================================================
 Chapter 12 案例闯关 · GRPO 强化学习（5 关，核心逻辑可跑可测）
================================================================================
 用法：
   python3 Chapter12_案例闯关_GRPO实战.py            # 看菜单
   python3 Chapter12_案例闯关_GRPO实战.py 3          # 只跑第 3 关
   python3 Chapter12_案例闯关_GRPO实战.py all         # 全部

 关卡：
   1  组内优势计算      不同奖励分布 → 优势(GRPO 的心脏，纯张量秒出)
   2  奖励函数工坊      长度/格式/正确性/XML 四种奖励，同一批补全对比
   3  裁剪损失的作用    看 clip 怎么限制“单步别更新太猛”
   4  端到端一步 GRPO   真实模型采一组补全 → 打分 → 组内优势(不反向,秒级)
   5  完整训练          指向 Chapter12_GRPO微调实战.py(真训练,看答对率上升)

 说明:1/2/3 纯 PyTorch/纯逻辑秒出;4 关下 SmolLM2-135M 采样;完整训练走第 5 关的实战文件。
================================================================================
"""

import re
import sys
import torch


def title(n, text):
    print("\n" + "=" * 72 + f"\n  第 {n} 关：{text}\n" + "=" * 72)


# ==============================================================================
# 第 1 关：组内优势计算
# ==============================================================================
def case_1():
    title(1, "组内优势 advantage = (r - 组均值)/组标准差")

    def advantages(rewards, G):
        g = rewards.view(-1, G)
        mean = g.mean(1).repeat_interleave(G)
        std = g.std(1).repeat_interleave(G)
        return (rewards - mean) / (std + 1e-8)

    for name, r in [("有对有错(1,0,0,1)", torch.tensor([1., 0., 0., 1.])),
                    ("全对(1,1,1,1)", torch.tensor([1., 1., 1., 1.])),
                    ("连续奖励(3,1,4,1)", torch.tensor([3., 1., 4., 1.]))]:
        adv = advantages(r, 4)
        print(f"  {name:18} → 优势 {[round(x,2) for x in adv.tolist()]}")
    print("\n  要点：全对(方差=0)时优势全≈0 —— 没有‘谁更好’就没有学习信号，")
    print("        所以 GRPO 要选‘组内有对有错’的题(奖励有方差)才学得动。")


# ==============================================================================
# 第 2 关：奖励函数工坊
# ==============================================================================
def case_2():
    title(2, "四种奖励函数对比")
    comps = [
        "<think>2+3=5</think><answer>5</answer>",   # 格式对+答案对
        "The answer is 5, obviously.",               # 无格式,答案对
        "<think>x</think><answer>7</answer>",        # 格式对,答案错
        "I don't know.",                             # 都不满足
    ]
    ans = ["5"] * 4

    def r_len(cs, ideal=25): return [-abs(ideal - len(c)) for c in cs]
    def r_fmt(cs): return [1.0 if re.search(r"<answer>.*?</answer>", c) else 0.0 for c in cs]
    def r_correct(cs, a): return [1.0 if x in c else 0.0 for c, x in zip(cs, a)]
    def r_xml(cs):
        def cnt(t): return sum(0.25 for tag in ["<think>","</think>","<answer>","</answer>"] if t.count(tag)==1)
        return [cnt(c) for c in cs]

    print("  补全:", [c[:32] for c in comps])
    print("  长度奖励  :", r_len(comps))
    print("  格式奖励  :", r_fmt(comps))
    print("  正确性奖励:", r_correct(comps, ans))
    print("  XML结构   :", r_xml(comps))
    print("\n  要点：GRPO 常把多个奖励加权相加(格式+正确性+简洁)，塑造模型的完整行为。")


# ==============================================================================
# 第 3 关：裁剪损失的作用
# ==============================================================================
def case_3():
    title(3, "裁剪(clip)：限制单步策略别更新太猛")

    def loss_at(ratio, adv, eps=0.2):
        r = torch.tensor([ratio]); a = torch.tensor([adv])
        unclipped = r * a
        clipped = torch.clamp(r, 1 - eps, 1 + eps) * a
        return -torch.min(unclipped, clipped).item()

    print("  优势=+1(想加大概率)时，概率比 ratio 越大 loss 越负；但 clip 到 1.2 就封顶：")
    for ratio in [1.0, 1.1, 1.2, 1.5, 2.0]:
        print(f"   ratio={ratio:<4} → loss={loss_at(ratio, 1.0):+.3f}  {'(被 clip 封顶)' if ratio>1.2 else ''}")
    print("\n  要点：不 clip 的话,优势为正会无脑把概率往上推、一步跳太远训练崩;")
    print("        clip 到 [1-ε,1+ε] 让每步只小改一点,更稳(这就是 PPO/GRPO 的稳定性来源)。")


# ==============================================================================
# 第 4 关：端到端一步 GRPO（真实模型，不反向）
# ==============================================================================
def case_4():
    title(4, "真实模型采一组补全 → 打分 → 组内优势")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  需要 transformers"); return
    name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name).eval()

    q, answer = "What is 2 plus 3? Answer with the number.", "5"
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt")
    G = 6
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=24, do_sample=True, temperature=1.0,
                             top_k=50, num_return_sequences=G, pad_token_id=tok.pad_token_id)
    texts = [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen]
    rewards = torch.tensor([1.0 if answer in t else 0.0 for t in texts])
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    print(f"  问题: {q}  (正确答案={answer})\n")
    for i, (t, r, a) in enumerate(zip(texts, rewards.tolist(), adv.tolist())):
        print(f"   [{i}] 奖励={r:.0f} 优势={a:+.2f} | {t.strip()[:50]!r}")
    print(f"\n  这一组答对率={rewards.mean()*100:.0f}%。GRPO 会强化优势>0(答对)的、抑制<0 的。")
    print("  完整训练(多步迭代看答对率上升)见 Chapter12_GRPO微调实战.py。")


# ==============================================================================
# 第 5 关：完整训练（指向实战文件）
# ==============================================================================
def case_5():
    title(5, "完整 GRPO 训练闭环")
    print("  完整的‘手写 GRPO 训练循环 + LoRA + 真训练看答对率上升’在:")
    print("     Chapter12_GRPO微调实战.py")
    print("  跑法: python3 Chapter12_GRPO微调实战.py --steps 40")
    print("  它做的:采样组→打分(可验证奖励)→组内优势→裁剪损失+KL→更新 LoRA,多步迭代。")
    print("  生产上云端用 trl.GRPOTrainer(+vLLM 加速采样),本机手写版证明你真懂算法。")


CASES = {i: globals()[f"case_{i}"] for i in range(1, 6)}


def run(choice):
    if choice == "all":
        for i in sorted(CASES):
            CASES[i]()
    elif choice.isdigit() and int(choice) in CASES:
        CASES[int(choice)]()
    else:
        print(f"没有第 {choice} 关，可选 {sorted(CASES)} 或 all")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
