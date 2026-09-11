"""
================================================================================
 Deep RL Course · Chapter 8 · 从零手写 PPO（学习笔记 · 裁剪目标 numpy 可跑）
================================================================================
 一句话：PPO 是最常用的策略优化——用"裁剪(clip)"限制每步策略更新幅度,稳定又好用(和 RLHF 的 PPO 同源)。
 本章讲(纯 numpy 实现 PPO 裁剪目标,看清 clip 怎么防更新过猛,无需 gym)：
   ① 概率比 ratio = π_new(a)/π_old(a);优势 A 衡量该动作比平均好多少。
   ② 裁剪目标:min(ratio·A, clip(ratio,1-ε,1+ε)·A) —— 当 ratio 偏离 1 太多就被截断。
   ③ Actor-Critic + GAE(见 ch9)。对照 LLM 的 GRPO(实战练习/Ch12)。
 要点：裁剪目标是 PPO 的灵魂——限制新旧策略别差太多,防为刷优势而更新过猛崩掉。
 跑：python3 chapter8_PPO从零实现_学习笔记.py   （裁剪目标纯 numpy 真跑)
================================================================================
"""
import numpy as np


def ppo_clip_objective(ratio, adv, eps=0.2):
    """PPO 目标(取负做 loss)：min(未裁剪, 裁剪后)。逐样本取更保守的那个。"""
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1 - eps, 1 + eps) * adv
    return np.minimum(unclipped, clipped)


def main():
    adv = np.array([1.0, 1.0, 1.0])
    ratio = np.array([1.0, 1.5, 0.5])                 # 1.5/0.5 都偏离 1 较多
    obj = ppo_clip_objective(ratio, adv, eps=0.2)
    # adv>0 且 ratio=1.5 → 被裁到 1.2(防更新过猛);ratio=1.0 不变
    assert np.isclose(obj[0], 1.0) and np.isclose(obj[1], 1.2)
    print(f"✅ Ch8 跑通：PPO 裁剪目标 ratio={ratio.tolist()} → {obj.round(2).tolist()}(1.5 被裁到 1.2)")
    # 面试：Q PPO 灵魂? A 裁剪目标限制新旧策略比值,防更新过猛; Q 和 GRPO 关系? A 同源,GRPO 用组内优势去掉 critic。


if __name__ == "__main__":
    main()
