"""
================================================================================
 案例4 · GRPO 强化学习微调（整合 Ch12 + Ch11）—— 完整可跑，照着手写练熟
================================================================================
 目标：用 GRPO(DeepSeek-R1 那套) + LoRA，靠“答对才给分”把模型往“更常答对”推。
 整合了哪几章、每步为什么用这个技术：
   [Ch12 GRPO]      为什么用它——对同一题采“一组”答案，谁比组内平均好就强化谁(组内相对优势)，
                    不需要单独的价值网络(比 PPO 简单)。
   [Ch12 可验证奖励] 为什么用“答案对不对”当奖励——数学/事实类任务答案能自动验证，不用训奖励模型；
                    这正是 R1 训推理能力的关键。
   [Ch12 组内优势]   advantage=(r-组均值)/组标准差；必须选“组内有对有错(有方差)”的题才有学习信号。
   [Ch11 LoRA]      为什么套 LoRA——RL 训练本就采样密集很贵，用 LoRA 只训 <1% 参数，Mac 也扛得住；
                    参考策略(算 KL)直接“关掉 LoRA 的基座”，优雅。

 直接运行：python3 案例4_GRPO强化学习.py     # 真训练(默认 12 步)，看答对率往上走
   (mps 上采样较慢是主要耗时；这是 GRPO 本身的特性，生产用 vLLM 加速采样。)
================================================================================
"""
import argparse
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
# 可验证的简单事实题：基座有时对有时错 → 组内有方差 → 有学习信号
QA = [
    ("What is 2 plus 3? Answer with the number.", "5"),
    ("What is the capital of France?", "Paris"),
    ("What color do you get by mixing blue and yellow?", "green"),
    ("How many legs does a spider have?", "8"),
]


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def reward(texts, answer):
    # 可验证奖励：补全里出现正确答案就 1 分，否则 0 分
    return torch.tensor([1.0 if answer.lower() in t.lower() else 0.0 for t in texts])


def composite_reward(texts, answer):
    """[知识点·奖励设计] 生产里奖励很少是单一“对/错”，而是多个信号加权组合：
      · 正确性(可验证)：答案对不对——最硬的信号(数学/代码/事实能自动判)。
      · 格式奖励：是否按要求格式(如 <think>...</think><answer>...</answer>)——R1 就用它逼出推理过程。
      · 简洁/长度：别啰嗦也别偷懒(可惩罚过长/过短)。
    组合奖励能引导“既答对、又按规范答”。但要小心 reward hacking:奖励设计有漏洞时，模型会钻空子
    刷分而非真的变好(如只输出格式空壳、或复读答案词)——所以正确性通常占主导、其余是小额加成。"""
    out = []
    for t in texts:
        r = 1.0 if answer.lower() in t.lower() else 0.0            # 正确性(主)
        r += 0.2 if 0 < len(t.split()) <= 12 else 0.0             # 简洁加成(次)
        out.append(r)
    return torch.tensor(out)


def demo_group_variance():
    """[知识点·组内方差为什么重要] advantage=(r-组均值)/组标准差。
    若一组答案全对(或全错)→ 方差=0 → 优势全 0 → 没有“谁更好”的信号 → 这条样本学不动。
    所以要挑“基座有对有错(组内有方差)”的题;全会/全不会的题对 GRPO 没有梯度贡献。"""
    def adv(r):
        r = torch.tensor(r)
        return (r - r.mean()) / (r.std() + 1e-8)
    # [自检] 组内方差演示（噪音，已静音）：优势 = (奖励-组均值)/组标准差
    #   有对有错 r=[1,0,1,0]→优势≈[1,-1,1,-1](有信号,对的被强化/错的被压)；
    #   全对 r=[1,1,1,1] / 全错 r=[0,0,0,0]→方差0→优势全≈0(这题学不动,白采样)
    # 顺带演示组合奖励:同样命中答案,简洁的那条多拿 0.2 分
    cr = composite_reward(
        ["Paris",
         "Well the capital city of the country known as France is of course the city named Paris"], "Paris")
    # [自检] 组合奖励演示（噪音，已静音）：都答对时简洁版多 +0.2 → 约 [1.2, 1.0]


def seq_logprob(model, ids, attn, comp_mask):
    """补全部分的总对数概率(t 位预测 t+1 位)。"""
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1, :]
    logp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (logp * comp_mask[:, 1:].float()).sum(1)


def main(cfg):
    device = pick_device()
    # [自检] 设备/模型/组大小 配置回显（噪音，已静音）
    demo_group_variance()                       # [知识点] 先看清“组内方差=学习信号”
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL)
    # [Ch11] LoRA：只训低秩适配器；参考策略=关掉 LoRA 的基座
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM", lora_dropout=0.0)).to(device)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)

    def build_prompt(q):
        return tok.apply_chat_template([{"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True)

    hist = []
    for step in range(cfg.steps):
        model.eval()
        seqs, cmasks, advs, rewards = [], [], [], []
        for q, a in QA:
            enc = tok(build_prompt(q), return_tensors="pt").to(device)
            plen = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=cfg.max_new, do_sample=True,
                                     temperature=1.0, top_k=50,
                                     num_return_sequences=cfg.group, pad_token_id=tok.pad_token_id)
            texts = [tok.decode(g[plen:], skip_special_tokens=True) for g in gen]
            r = reward(texts, a)
            advs.append((r - r.mean()) / (r.std() + 1e-8))      # [Ch12] 组内优势
            rewards.append(r)
            for g in gen:
                m = torch.zeros_like(g); m[plen:] = 1
                seqs.append(g); cmasks.append(m)
        advantages = torch.cat(advs).to(device)
        hist.append(torch.cat(rewards).mean().item())

        # 右侧 pad 到同长，堆 batch
        L = max(s.shape[0] for s in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id, device=device)
        attn = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
        comp = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
        for i, (s, c) in enumerate(zip(seqs, cmasks)):
            ids[i, :s.shape[0]] = s; attn[i, :s.shape[0]] = 1; comp[i, :c.shape[0]] = c

        model.train()
        policy = seq_logprob(model, ids, attn, comp)             # 带梯度
        with torch.no_grad(), model.disable_adapter():           # 参考=基座(无 LoRA)
            ref = seq_logprob(model, ids, attn, comp)
        # loss 两项：①-(优势×策略对数概率)：优势>0 的补全被抬高概率(强化),优势<0 的被压低。
        #   ②cfg.beta*(ref-policy)：KL 惩罚，拉住策略别离“参考模型(基座)”太远。
        # [知识点·KL 惩罚作用] 只追奖励会让模型为刷分崩坏(输出退化/胡言/reward hacking)、也会忘掉
        #   基座的通用能力;KL 项像根皮筋把它拴在参考策略附近，稳住训练。beta 太大学不动、太小易崩。
        loss = -(advantages * policy).mean() + cfg.beta * (ref - policy).mean()  # 策略梯度+KL
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 3 == 0 or step == cfg.steps - 1:
            print(f"   step {step:2}: 答对率={hist[-1]*100:5.1f}%  loss={loss.item():.3f}")

    early, late = sum(hist[:2]) / 2, sum(hist[-2:]) / 2
    print(f"\n>>> 答对率：前2步={early*100:.0f}% → 后2步={late*100:.0f}%  "
          f"({'↑ 上升，GRPO 在起作用' if late > early else '波动(RL 噪声大，多训几步/更大模型更稳)'})")
    # print(">>> GRPO 跑通：采样组→可验证奖励→组内优势(Ch12)→LoRA 更新(Ch11)。云端用 trl.GRPOTrainer+vLLM。")


# ==============================================================================
# 面试题(GRPO / RLHF)
# ==============================================================================
# Q1: GRPO 和 PPO 的核心区别？GRPO 为什么更省？
# A : PPO 要额外训一个价值网络(critic)估基线，占显存又难调;GRPO 用“同一题采一组答案、组内相对好坏”
#     当优势(基线=组均值)，去掉了 critic，更简单省显存。DeepSeek-R1 用 GRPO 训出推理能力。
# Q2: 组内优势怎么算？为什么要有组内方差？
# A : advantage=(reward-组均值)/组标准差,只比“同题里谁更好”,天然消掉题目难度差异。若一组全对/全错→
#     方差0→优势全≈0→无学习信号。所以要挑基座“有对有错”的题,全会/全不会的对训练没贡献。
# Q3: 为什么用“可验证奖励”？还常配哪些奖励？
# A : 数学/代码/事实题答案能自动判对错(或跑单测/能编译),直接拿正确性当奖励最可靠,不用训奖励模型。
#     常再加格式奖励(逼出<think>推理)、长度/简洁奖励等，多个加权组合。
# Q4: KL 惩罚项的作用？beta 太大/太小会怎样？
# A : 拉住新策略别离参考模型(基座)太远,防为刷奖励而输出退化、遗忘通用能力。beta 太大=被拴太死学不动;
#     太小=容易崩坏(reward hacking/胡言)。监控 KL 别爆。
# Q5: 什么是 reward hacking？怎么防？
# A : 模型钻奖励函数的漏洞刷高分但没真变好(如只堆格式空壳、复读答案词、答非所问但触发关键词)。
#     防:正确性占主导、奖励尽量可验证、加 KL 约束、多信号交叉、人工抽检、对拒答/空壳单独惩罚。


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--max_new", type=int, default=24)
    p.add_argument("--beta", type=float, default=0.01)
    main(p.parse_args())
