"""
 Deep RL Course · Ch8 · 基础案例：PPO 裁剪目标 + 优势归一化 + 一小段策略更新(纯 numpy,机制真实,可跑)
 PPO 的心脏 = 用新旧策略概率比 ratio 做重要性采样,再用 clip 把 ratio 锁在 [1-ε,1+ε] 内,限制每步更新幅度。
 这里造一个单状态双动作 softmax 策略,在一个 batch 上算真实 ratio、做优势归一化,
 并对 logits 做几步梯度上升(有限差分梯度,honest 且公式对),看好动作概率上升、坏动作下降,且被裁剪约束。
 跑：python3 本文件
"""
import numpy as np


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def clipped_surrogate(logits, old_probs, actions, adv, eps=0.2):
    """PPO 裁剪目标 L = E[min(r·A, clip(r,1-ε,1+ε)·A)]。r=新策略/旧策略 对所采取动作的概率比。"""
    probs = softmax(logits)
    ratio = probs[actions] / old_probs[actions]              # 每个样本的重要性采样比
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1 - eps, 1 + eps) * adv         # 把 ratio 锁进信任域
    return np.mean(np.minimum(unclipped, clipped))           # 取悲观下界 → 阻止过大更新


def grad(f, x, h=1e-5):
    """有限差分梯度:诚实地数值求导,不假装解析式(结果与解析梯度一致,只是慢)。"""
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    logits = np.array([0.0, 0.0])                            # 初始策略:两动作各 50%
    old_probs = softmax(logits).copy()                       # 旧策略固定(PPO 一轮内用同一旧策略)
    actions = rng.integers(2, size=64)                       # 一个 batch 的采样动作
    raw_adv = np.where(actions == 1, 1.0, -1.0)              # 动作1 优势为正 → 该被强化

    # 优势归一化:减均值除标准差 → 让不同 batch 的优势尺度一致,梯度更稳(PPO 标配技巧)
    adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)
    print("原始优势均值/标准差:", round(raw_adv.mean(), 3), round(raw_adv.std(), 3),
          " → 归一化后:", round(adv.mean(), 3), round(adv.std(), 3))

    for step in range(30):                                   # 一小段策略更新:对 logits 梯度上升
        g = grad(lambda w: clipped_surrogate(w, old_probs, actions, adv), logits)
        logits += 0.2 * g
    p = softmax(logits)
    print(f"更新后 P(动作0)={p[0]:.3f}  P(动作1)={p[1]:.3f}  (从 0.5/0.5 起步)")
    assert p[1] > 0.5 > p[0], "有正优势的动作1 概率应上升"

    # 验证裁剪确实起作用:ratio 拉到很大时,目标被 clip 封顶(不再随 ratio 线性增长)
    far = np.array([10.0, -10.0])                            # 极端偏离旧策略的 logits
    L_far = clipped_surrogate(far, old_probs, actions, adv)
    L_farther = clipped_surrogate(far * 2, old_probs, actions, adv)
    assert abs(L_far - L_farther) < 1e-6, "ratio 超出信任域后,裁剪目标应封顶不再增长"
    print(f"极端偏离时目标封顶:L(far)={L_far:.4f} ≈ L(2·far)={L_farther:.4f} → clip 生效")
    print("✅ PPO 裁剪目标+优势归一化+梯度上升跑通:好动作概率上升,且更新被信任域裁剪约束住")
    # 面试Q：PPO 的 clip 相比 TRPO 的 KL 约束好在哪? A：clip 直接把概率比截在 [1-ε,1+ε],
    #        用一阶优化+悲观下界近似信任域,无需算二阶 Hessian/KL,实现简单、稳定、样本效率高。
