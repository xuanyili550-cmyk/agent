import re
import torch

# 设 B=2 组、每组 G=4 个答案，共 8 个补全，奖励如下(1=好,0=差)：
rewards = torch.tensor([1., 0., 0., 1.,   0., 0., 1., 1.])
G = 4
grouped = rewards.view(-1, G)                     # (B, G) = (2, 4)
mean = grouped.mean(dim=1).repeat_interleave(G)   # 每组均值，广播回 8 个
std = grouped.std(dim=1).repeat_interleave(G)     # 每组标准差，广播回 8 个
advantages = (rewards - mean) / (std + 1e-8)      # 组内相对优势

def reward_length(completions,ideal=50,**kw):
    return [-abs(ideal-len(c)) for c in completions]
def reward_format(completions, **kw):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    return [1.0 if re.match(pattern, c, re.DOTALL) else 0.0 for c in completions]
def reward_correct(completions, answers, **kw):
    def extract(c):
        m = re.search(r"<answer>(.*?)</answer>", c, re.DOTALL)
        return m.group(1).strip() if m else ""
    return [1.0 if extract(c) == a else 0.0 for c, a in zip(completions, answers)]

def count_xml(text):
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
for i, o in enumerate(out):
    gen = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)

