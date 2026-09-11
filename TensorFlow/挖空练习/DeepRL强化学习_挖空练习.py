"""
================================================================================
 Deep RL 挖空练习 · Q-Learning / PPO 裁剪 / GAE（对应 Deep RL Course）
================================================================================
 玩法：遮住「练习N」下面那行默写,再运行对照。python3 DeepRL强化学习_挖空练习.py (纯 numpy)
================================================================================
"""
import numpy as np

# —— 练习1：Q-Learning 的 TD 更新 ——
def td_update(Q, s, a, r, s2, alpha=0.5, gamma=0.9):
    Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])   # ← 练习1:r+γ·maxQ(s')−Q(s,a)
    return Q

# —— 练习2：PPO 裁剪目标 ——
def ppo_clip(ratio, adv, eps=0.2):
    return np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)   # ← 练习2:min(未裁,裁剪)

# —— 练习3：GAE 优势(从后往前累加) ——
def gae(r, v, gamma=0.99, lam=0.95):
    adv, g = np.zeros(len(r)), 0.0
    for t in reversed(range(len(r))):
        delta = r[t] + gamma * v[t + 1] - v[t]               # ← 练习3:TD 误差 δ
        g = delta + gamma * lam * g
        adv[t] = g
    return adv


def main():
    Q = np.zeros((3, 2)); Q[1] = [0.2, 0.8]
    td_update(Q, 0, 1, 0.0, 1)
    assert abs(Q[0, 1] - 0.5 * 0.9 * 0.8) < 1e-9
    assert abs(ppo_clip(np.array([1.5]), np.array([1.0]))[0] - 1.2) < 1e-9   # 1.5 裁到 1.2
    a = gae(np.array([1.0, 1, 1]), np.array([0.5, 0.5, 0.5, 0.0]))
    assert a.shape == (3,)
    print("✅ Deep RL 挖空全部通过:TD 更新 + PPO 裁剪 + GAE")


if __name__ == "__main__":
    main()

# ================================ 答案要点 ====================================
# 练习1: Q[s,a] += alpha*(r + gamma*Q[s2].max() - Q[s,a])
# 练习2: np.minimum(ratio*adv, np.clip(ratio,1-eps,1+eps)*adv)
# 练习3: delta = r[t] + gamma*v[t+1] - v[t]; g = delta + gamma*lam*g
# ============================================================================
