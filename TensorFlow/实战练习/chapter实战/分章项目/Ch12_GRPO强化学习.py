"""
================================================================================
 分章项目 · Ch12 · GRPO 强化学习（贴 HF Ch12：组内优势 + 可验证奖励 + 裁剪损失 + 多奖励塑形）
================================================================================
 HF 课程 Ch12“构建推理模型(GRPO，DeepSeek-R1 那套)”的核心四件，纯张量秒出、带自检：
   ① 组内相对优势：对同一题采一组答案，谁比组内平均好就强化(advantage=(r-均值)/标准差)。
   ② 可验证奖励：答案对不对自动判(数学/代码/事实)，不用训奖励模型——R1 训推理的关键。
   ③ 裁剪策略损失 + KL：优势为正就加大概率，但 clip 限制单步别跳太猛(和 PPO 一样)。
   ④ 多奖励塑形：正确性 + 格式等多个奖励加权组合，塑造完整行为(真实 GRPO 就这么设)。
   完整“真训练看答对率上升”的手写版见 ../案例4_GRPO强化学习.py；生产(trl.GRPOTrainer+vLLM)见 ../生产架构/生产04。
 跑：python3 Ch12_GRPO强化学习.py
================================================================================
"""
import re
import torch


# ==============================================================================
# ① 组内相对优势
# ==============================================================================
def group_advantage():
    # print("=" * 70, "\n① 组内相对优势(同题一组答案，比组内平均)\n" + "=" * 70)
    rewards = torch.tensor([1., 0., 0., 1., 0., 0., 1., 1.]); G = 4
    g = rewards.view(-1, G)
    adv = (rewards - g.mean(1).repeat_interleave(G)) / (g.std(1).repeat_interleave(G) + 1e-8)
    print("  组内优势:", [round(x, 2) for x in adv.tolist()])
    assert abs(adv.view(-1, G).sum(1)).max() < 1e-4          # 每组优势和≈0
    # print("  正=强化(比组内平均好)、负=抑制；全对/全错则优势全0=无信号，所以要选“有对有错”的题。")
    return adv


# ==============================================================================
# ② 可验证奖励
# ==============================================================================
def verifiable_reward():
    # print("\n" + "=" * 70, "\n② 可验证奖励(答案对不对自动判)\n" + "=" * 70)
    def reward_correct(comps, ans):
        def ext(c):
            m = re.search(r"<answer>(.*?)</answer>", c); return m.group(1).strip() if m else ""
        return [1.0 if ext(c) == ans else 0.0 for c in comps]
    comps = ["<think>2+3=5</think><answer>5</answer>", "the answer is 5", "<answer>4</answer>"]
    r = reward_correct(comps, "5")
    print("  三个答案(标准答案=5)的正确性奖励:", r)
    assert r == [1.0, 0.0, 0.0]
    # print("  可验证=不用训奖励模型；数学/代码/事实题答案能自动对错，信号最可靠。")


# ==============================================================================
# ③ 裁剪策略损失 + KL
# ==============================================================================
def grpo_loss_demo(adv):
    # print("\n" + "=" * 70, "\n③ 裁剪策略损失 + KL 惩罚(和 PPO 一样)\n" + "=" * 70)
    def grpo_loss(new_lp, old_lp, ref_lp, adv, eps=0.2, beta=0.04):
        ratio = torch.exp(new_lp - old_lp)
        a = adv.unsqueeze(1)
        policy = -torch.min(ratio * a, torch.clamp(ratio, 1 - eps, 1 + eps) * a)   # 裁剪
        kl = torch.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1                     # 与参考模型的 KL
        return (policy + beta * kl).mean()
    torch.manual_seed(0)
    old = torch.randn(8, 1); new = old + 0.1 * torch.randn(8, 1)
    loss = grpo_loss(new, old, old.clone(), adv).item()
    print(f"  GRPO 损失 = {loss:.4f}")
    # print("  clip(1±ε) 限制单步更新幅度别跳太猛；KL 拉住策略别离参考模型太远(否则语言崩掉)。")


# ==============================================================================
# ④ 多奖励塑形：正确性 + 格式，加权组合
# ==============================================================================
def multi_reward_shaping():
    # print("\n" + "=" * 70, "\n④ 多奖励塑形(正确性 + 格式，加权组合)\n" + "=" * 70)
    def correctness(comps, ans):
        def ext(c):
            m = re.search(r"<answer>(.*?)</answer>", c); return m.group(1).strip() if m else ""
        return [2.0 if ext(c) == ans else 0.0 for c in comps]      # 权重高
    def format_ok(comps):
        pat = r"<think>.*?</think>\s*<answer>.*?</answer>"
        return [0.5 if re.search(pat, c, re.DOTALL) else 0.0 for c in comps]   # 权重低
    comps = ["<think>2+3=5</think><answer>5</answer>",   # 又对又规范 → 2.0+0.5
             "<answer>5</answer>",                       # 对但没思考过程 → 2.0+0.0
             "<think>hmm</think><answer>4</answer>"]     # 规范但答错 → 0.0+0.5
    c, f = correctness(comps, "5"), format_ok(comps)
    total = [a + b for a, b in zip(c, f)]
    for i, comp in enumerate(comps):
        print(f"  正确={c[i]} 格式={f[i]} 合计={total[i]}  ← {comp[:40]}")
    assert total == [2.5, 2.0, 0.5]
    # print("  多奖励组合塑造完整行为(答对+按格式推理)；要防“奖励黑客”(模型钻某个奖励的空子)。")


if __name__ == "__main__":
    adv = group_advantage()
    verifiable_reward()
    grpo_loss_demo(adv)
    multi_reward_shaping()
    print("\n✅ Ch12 跑通：组内优势 + 可验证奖励 + 裁剪损失/KL + 多奖励塑形。真训练看答对率上升见 ../案例4。")
    # print("面试：Q GRPO vs PPO? Q 优势怎么算?为什么要组内有方差? Q 为什么用可验证奖励? Q 多奖励怎么组合?"
          # " (见 ../面试高频题库.py 六)")
