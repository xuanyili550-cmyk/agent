"""
 Deep RL Course · Ch9 · 基础案例：GAE 完整反向递归 + 不同 λ 的偏差/方差权衡(纯 numpy,机制真实,可跑)
 GAE(广义优势估计)= 对多步 TD 误差 δ_t 做 γλ 指数加权和,用一个 λ 旋钮在偏差与方差间连续调节。
 这里实现带 done 掩码的完整反向递归,并验证两个理论端点:
   λ=0 → 优势=单步 TD(δ_t),低方差高偏差; λ=1 → 优势=蒙特卡洛回报-基线,低偏差高方差。
 跑：python3 本文件
"""
import numpy as np


def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """完整 GAE:反向递归 A_t = δ_t + γλ(1-done)·A_{t+1},δ_t = r_t + γ(1-done)V_{t+1} - V_t。
    values 长度比 rewards 多 1(末尾是 bootstrap 的 V(s_T));done=1 时切断未来,不跨 episode 泄漏。"""
    T = len(rewards)
    adv, gae_acc = np.zeros(T), 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]                          # 终止步后未来清零
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae_acc = delta + gamma * lam * nonterminal * gae_acc  # γλ 指数加权累积
        adv[t] = gae_acc
    return adv


def mc_advantage(rewards, values, dones, gamma=0.99):
    """蒙特卡洛优势:实际折扣回报 G_t 减基线 V_t(GAE 在 λ=1 时应退化到这个)。"""
    T, adv, g = len(rewards), np.zeros(len(rewards)), 0.0
    for t in reversed(range(T)):
        g = rewards[t] + gamma * g * (1.0 - dones[t])
        adv[t] = g - values[t]
    return adv


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rewards = rng.normal(1.0, 0.5, size=8)                    # 一条轨迹的奖励
    values = rng.normal(0.5, 0.1, size=9)                     # V(s_0..s_8),末位为 bootstrap
    dones = np.zeros(8); dones[-1] = 1.0                       # 最后一步 episode 结束

    a0 = gae(rewards, values, dones, lam=0.0)                 # λ=0
    a95 = gae(rewards, values, dones, lam=0.95)               # 常用中间值
    a1 = gae(rewards, values, dones, lam=1.0)                 # λ=1

    # 端点验证① λ=0 时 A_t 就是单步 TD 误差 δ_t
    td = rewards + 0.99 * values[1:] * (1 - dones) - values[:-1]
    assert np.allclose(a0, td), "λ=0 的 GAE 应等于单步 TD 误差"
    # 端点验证② λ=1 时 A_t 等于蒙特卡洛优势(折扣回报-基线)
    assert np.allclose(a1, mc_advantage(rewards, values, dones)), "λ=1 的 GAE 应等于 MC 优势"

    print("λ=0.00 (TD, 低方差高偏差) 优势:", a0.round(3))
    print("λ=0.95 (常用折中)         优势:", a95.round(3))
    print("λ=1.00 (MC, 低偏差高方差) 优势:", a1.round(3))
    print(f"优势方差随 λ 增大:var(λ=0)={a0.var():.3f} < var(λ=1)={a1.var():.3f}")
    assert a0.var() < a1.var(), "λ 越大越偏 MC → 方差越大(直觉验证)"
    print("✅ 带 done 的 GAE 反向递归跑通,λ=0/1 两端点与 TD/MC 理论一致 → λ 是偏差-方差的连续旋钮")
    # 面试Q：GAE 里 λ 起什么作用? A：λ 在单步 TD(λ=0,偏差大方差小)和蒙特卡洛(λ=1,偏差小方差大)之间插值,
    #        对多步 TD 误差做几何加权;λ≈0.95 常能兼顾稳定与准确,是 PPO 优势估计的默认选择。
