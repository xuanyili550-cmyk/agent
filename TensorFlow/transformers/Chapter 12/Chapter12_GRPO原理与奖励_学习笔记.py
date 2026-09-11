"""
================================================================================
 Chapter 12 · GRPO 强化学习(RL for LLMs) —— 原理与奖励函数 学习笔记（可运行）
================================================================================
 配套「Reinforcement Learning for LLMs」章。GRPO(Group Relative Policy Optimization)
 是 DeepSeek-R1 用来训练“会推理”的模型的核心算法，现在是最热的 LLM 对齐/推理方法之一。

 ★ GRPO 一句话：对同一个问题让模型生成“一组(group)”答案，谁比组内平均好就强化谁、
   差就抑制谁(组内相对优势)。相比 PPO 它不需要单独的价值网络(critic)，更省、更简单。

 本文件把 GRPO 里“能在 Mac 上秒跑、又是核心”的逻辑都做成可运行 + 自检：
   1  GRPO vs PPO/RLHF：它在整个对齐家族里的位置(文字)
   2  ★组内优势归一化：rewards → (r-mean)/std(这是 GRPO 的心脏)
   3  ★奖励函数大全：长度/格式(正则)/答案正确性/XML 结构(都可跑可测)
   4  ★裁剪策略损失 + KL 惩罚：给定新旧 logprob 怎么算 loss
   5  从真实模型采一组补全(SmolLM2)：看“组”长什么样
   6  训练循环骨架 + trl GRPOTrainer 生产参考(文字)

 直接运行：python3 Chapter12_GRPO原理与奖励_学习笔记.py
 真训练闭环见同目录 Chapter12_GRPO微调实战.py。
================================================================================
"""

import re
import torch
import torch.nn.functional as F


def banner(t):
    print("\n" + "=" * 72 + f"\n {t}\n" + "=" * 72)


# ==============================================================================
# 1) GRPO 在对齐家族里的位置（文字）
# ==============================================================================
banner("1) GRPO 是什么、和 PPO/RLHF 的关系")
print("""
  对齐 LLM 的几条路线：
   · SFT(监督微调)      : 给“标准答案”让模型模仿(Ch11)。简单，但只会照抄示范。
   · RLHF/PPO           : 训一个奖励模型打分 + PPO 优化，还要一个价值网络(critic)。效果好但重、难调。
   · DPO                : 用“偏好对(chosen/rejected)”直接优化，免奖励模型，比 PPO 简单。
   · ★GRPO              : 对每个提示采“一组”答案，用“组内相对好坏”当优势，免价值网络。
                          DeepSeek-R1 用它 + 可验证奖励(答案对不对)训出了强推理能力。
  为什么 GRPO 适合“推理”：数学/代码这类任务答案可自动验证(对/错)，直接拿正确性当奖励，
  不用训奖励模型，配上“组内相对优势”就能把模型往“更常答对”的方向推。
""")


# ==============================================================================
# 2) ★组内优势归一化（GRPO 的心脏）
# ==============================================================================
banner("2) 组内优势：advantage = (reward - 组均值) / 组标准差")
# 设 B=2 组、每组 G=4 个答案，共 8 个补全，奖励如下(1=好,0=差)：
rewards = torch.tensor([1., 0., 0., 1.,   0., 0., 1., 1.])
G = 4
grouped = rewards.view(-1, G)                     # (B, G) = (2, 4)
mean = grouped.mean(dim=1).repeat_interleave(G)   # 每组均值，广播回 8 个
std = grouped.std(dim=1).repeat_interleave(G)     # 每组标准差，广播回 8 个
advantages = (rewards - mean) / (std + 1e-8)      # 组内相对优势
print("  各组奖励:", grouped.tolist())
print("  组均值(广播):", [round(x, 2) for x in mean.tolist()])
print("  优势 advantage:", [round(x, 2) for x in advantages.tolist()])
# 自检：优势 >0 表示“比组内平均好”(要强化)，<0 表示“比平均差”(要抑制)；每组优势和≈0
assert abs(advantages.view(-1, G).sum(dim=1)).max() < 1e-4
print("  ✅ 自检：每组优势之和≈0(相对量)；正=强化、负=抑制。")
print("  为什么这样算：不需要绝对分数，只看‘同一题里谁更好’，天然消掉题目难度差异。")


# ==============================================================================
# 3) ★奖励函数大全（GRPO 的“评分标准”，可跑可测）
# ==============================================================================
banner("3) 奖励函数：GRPO 靠它给每个补全打分")

def reward_length(completions, ideal=50, **kw):
    # 越接近理想长度奖励越高(负的绝对差)。用来控制输出长短。
    return [-abs(ideal - len(c)) for c in completions]

def reward_format(completions, **kw):
    # 结构奖励：必须是 <think>...</think><answer>...</answer> 才给分(教模型按格式输出)
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    return [1.0 if re.match(pattern, c, re.DOTALL) else 0.0 for c in completions]

def reward_correct(completions, answers, **kw):
    # 可验证奖励：抽出答案和标准答案比，对=1 错=0(数学/代码任务的核心，R1 就靠这个)
    def extract(c):
        m = re.search(r"<answer>(.*?)</answer>", c, re.DOTALL)
        return m.group(1).strip() if m else ""
    return [1.0 if extract(c) == a else 0.0 for c, a in zip(completions, answers)]

def count_xml(text):
    # XML 结构计数奖励：标签齐全给分，答案后有多余内容轻微扣分(鼓励干净格式)
    s = 0.0
    if text.count("<think>") == 1: s += 0.25
    if text.count("</think>") == 1: s += 0.25
    if text.count("<answer>") == 1: s += 0.25
    if text.count("</answer>") == 1: s += 0.25
    return s

comps = [
    "<think>2*2=4, +1=5</think><answer>5</answer>",   # 格式对、答案对
    "The answer is 5.",                                # 无格式
    "<think>hmm</think><answer>4</answer>",            # 格式对、答案错
]
answers = ["5", "5", "5"]
print("  reward_length(ideal=20):", reward_length(comps, ideal=20))
print("  reward_format          :", reward_format(comps))
print("  reward_correct         :", reward_correct(comps, answers))
print("  count_xml              :", [count_xml(c) for c in comps])
# 自检：第0条格式对+答案对；第1条全不满足；第2条格式对但答案错
assert reward_format(comps) == [1.0, 0.0, 1.0]
assert reward_correct(comps, answers) == [1.0, 0.0, 0.0]
print("  ✅ 自检：格式奖励认结构、正确性奖励认答案 —— GRPO 常把多个奖励加权组合。")


# ==============================================================================
# 4) ★裁剪策略损失 + KL 惩罚（给定新旧 logprob 算 loss）
# ==============================================================================
banner("4) GRPO 的损失：裁剪的策略梯度 + KL 约束")
# 直觉：ratio = 新策略概率 / 旧策略概率。优势为正就想加大 ratio(更爱这个答案)，
# 但用 clip 限制单步别变太猛(和 PPO 一样)；再加 KL 惩罚，别跑离参考模型太远。
def grpo_loss(new_logp, old_logp, ref_logp, advantages, eps=0.2, beta=0.04):
    ratio = torch.exp(new_logp - old_logp)                 # 概率比
    adv = advantages.unsqueeze(1)                          # (N,1) 广播到每个 token
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    policy_loss = -torch.min(unclipped, clipped)           # 取更保守的那个(裁剪)
    # KL(新||参考)：别让模型为了刷奖励跑偏、崩掉语言能力
    kl = torch.exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1   # 无偏 KL 估计
    return (policy_loss + beta * kl).mean()

N = 8
torch.manual_seed(0)
old = torch.randn(N, 1)
new = old + 0.1 * torch.randn(N, 1)     # 新策略略微变化
ref = old.clone()
loss = grpo_loss(new, old, ref, advantages)
print(f"  给定 8 个补全的新/旧/参考 logprob + 优势 → GRPO loss = {loss.item():.4f}")
print("  三要素：① 裁剪的策略梯度(优势×概率比，clip 防跳变) ② KL 惩罚(别跑离参考)")
print("          ③ 对优势为正的补全加大概率、为负的减小 —— 这就是‘强化好答案’。")


# ==============================================================================
# 5) 从真实模型采一组补全（看“组”长什么样）
# ==============================================================================
banner("5) 从真实模型给一个提示采 4 个补全(GRPO 的‘组’)")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Give a one-word color."}],
        tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=8, num_return_sequences=4,
                         do_sample=True, temperature=0.9, top_k=50,
                         pad_token_id=tok.eos_token_id)
    print("  同一提示采样 4 个补全(GRPO 会给这 4 个各打分、算组内优势)：")
    for i, o in enumerate(out):
        gen = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"   [{i}] {gen.strip()!r}")
    print("  要点：do_sample+temperature 让一组答案有差异，才有‘谁比谁好’可比。")
except Exception as e:
    print("  (跳过：需要下 SmolLM2 模型)", type(e).__name__, e)


# ==============================================================================
# 6) 训练循环骨架 + trl 生产参考（文字）
# ==============================================================================
banner("6) GRPO 训练循环骨架 + 生产工具")
print('''
  每步 GRPO(伪代码)：
    for prompt in batch:
        comps = policy.generate(prompt, num=G)          # 采一组
        rewards = [reward_fn(c) for c in comps]         # 打分
        adv = (rewards - mean(rewards)) / std(rewards)  # 组内优势(第2节)
        loss = clipped_policy_loss(adv) + beta*KL(policy||ref)  # (第4节)
        loss.backward(); optimizer.step()
  → 真·可跑的手写版见 Chapter12_GRPO微调实战.py。

  生产标准工具：trl 的 GRPOTrainer(最省事)：
    from trl import GRPOConfig, GRPOTrainer
    trainer = GRPOTrainer(model=..., reward_funcs=[reward_len, reward_format],
                          args=GRPOConfig(num_generations=8, use_vllm=True, ...),
                          train_dataset=ds)  # ds 要有 "prompt" 列
    trainer.train()
  ⚠ 在你这台 Mac 上 GRPOTrainer 依赖 mergekit(未装) + 实战要 GPU/vLLM/bitsandbytes(CUDA-only)，
    所以本章的“真训练”用手写版(Mac 能跑)，GRPOTrainer 作为上云端 GPU 时的生产参考。
''')

print("✅ 全部跑完。GRPO 三大件(组内优势 / 奖励函数 / 裁剪损失+KL)都可跑可测。")
