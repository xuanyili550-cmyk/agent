"""
================================================================================
 Chapter 12 挖空练习 · GRPO 组内优势归一化（GRPO 的心脏）
================================================================================
 玩法：
   1) 把每个 ______ 换成正确代码（凭记忆，别翻笔记）
   2) 运行：python3 Chapter12_挖空练习.py   （纯 PyTorch 张量，秒出，不下模型）
   3) 没填的报 NameError；卡住 → 文件底部「答案区」
 目标：把“一组答案的奖励”变成“组内相对优势”—— 这是 GRPO 区别于 PPO 的核心一步。
       (完整 GRPO 训练闭环见 Chapter 12/Chapter12_GRPO微调实战.py)

 回顾：GRPO 对同一题采一组(G 个)答案，谁比组内平均好就强化谁。优势公式：
   advantage = (reward - 组内均值) / (组内标准差)
================================================================================
"""
import torch

# 设 2 组、每组 4 个答案，共 8 个补全的奖励(1=好, 0=差)
rewards = torch.tensor([1., 0., 0., 1., 0., 0., 1., 1.])
G = 4

# 练习1：把奖励按“组”reshape 成 (B, G)（每行一组）。用 .view，第一维填 -1 自动推断
grouped = rewards.view(-1, G)

# 练习2：每组的均值（在“组内”这一维求平均，dim=?），再广播回 8 个
#   提示：mean(dim=1) 得到每组一个值，再 .repeat_interleave(G) 复制回每个成员
mean = grouped.mean(dim=1).repeat_interleave(G)

# 练习3：每组的标准差（同理，dim=1），广播回 8 个
std = grouped.std(dim=1).repeat_interleave(G)

# 练习4：组内优势 = (奖励 - 组均值) / (组标准差 + 1e-8)  （+1e-8 防除零）
advantages = (rewards - mean) / (std + 1e-8)

print("各组奖励:", grouped.tolist())
print("组均值  :", [round(x, 2) for x in mean.tolist()])
print("优势    :", [round(x, 2) for x in advantages.tolist()])

# 自检：优势 >0 = 比组内平均好(要强化)，<0 = 比平均差(要抑制)；每组优势之和应≈0
assert abs(advantages.view(-1, G).sum(dim=1)).max() < 1e-4, "每组优势之和应≈0"
assert advantages[0] > 0 and advantages[1] < 0, "第0个(答对)应为正、第1个(答错)应为负"
print("\n自检通过 ✅：组内相对优势算对了——正的强化、负的抑制，天然消掉题目难度差异。")


# ==============================================================================
#  答案区（卡住再看！）
# ------------------------------------------------------------------------------
#  1: rewards.view(-1, G)
#  2: grouped.mean(dim=1).repeat_interleave(G)
#  3: grouped.std(dim=1).repeat_interleave(G)
#  4: (rewards - mean) / (std + 1e-8)
# ==============================================================================
