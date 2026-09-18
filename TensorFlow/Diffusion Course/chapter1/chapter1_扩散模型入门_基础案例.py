"""
 Diffusion Course · Ch1 · 基础案例：前向扩散加噪的真实机制(纯 numpy,可跑)
 扩散前向 = 一条马尔可夫链:x_t = √(1-β_t)·x_{t-1} + √β_t·z(每步加一点噪声)。
 它有个关键"闭式"性质:可以一步跳到任意 t —— x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε。
 本文件① 验证"逐步链式"与"闭式一步跳"在分布上等价(蒙特卡洛核对均值/方差),
 ② 展示信噪比 SNR=ᾱ/(1-ᾱ) 随 t 单调下降,③ 已知 ε 时反解回 x₀。跑：python3 本文件
"""
import numpy as np

T = 1000
betas = np.linspace(1e-4, 0.02, T)          # 线性噪声表(DDPM 原版调度)
alphas = 1.0 - betas
abar = np.cumprod(alphas)                    # ᾱ_t = ∏(1-β):累计"信号保留比例"
rng = np.random.default_rng(0)


def forward_closed(x0, t, eps):
    """闭式一步跳到 t:训练时就是这么算的(不用真的循环 t 步)。"""
    return np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps


def forward_stepwise(x0, t):
    """老实按马尔可夫链走 t 步,每步加独立噪声 —— 用来核对闭式是否等价。"""
    x = x0.copy()
    for k in range(t + 1):
        x = np.sqrt(alphas[k]) * x + np.sqrt(betas[k]) * rng.standard_normal(x.shape)
    return x


if __name__ == "__main__":
    x0 = np.ones(4) * 0.8                     # 干净信号(全 0.8)
    t = 200

    # ① 逐步链 vs 闭式:同一分布,蒙特卡洛核对(注意是分布等价,不是逐样本相等)
    N = 20000
    step_samples = np.stack([forward_stepwise(x0, t) for _ in range(400)])
    clsd_samples = np.stack([forward_closed(x0, t, rng.standard_normal(x0.shape)) for _ in range(N)])
    print(f"t={t}: 闭式  均值≈{clsd_samples.mean():+.3f}(理论√ᾱ·0.8={np.sqrt(abar[t])*0.8:+.3f})  std≈{clsd_samples.std():.3f}(理论√(1-ᾱ)={np.sqrt(1-abar[t]):.3f})")
    print(f"      逐步链均值≈{step_samples.mean():+.3f}  std≈{step_samples.std():.3f}  → 与闭式一致")
    assert abs(clsd_samples.mean() - np.sqrt(abar[t]) * 0.8) < 0.02
    assert abs(clsd_samples.std() - np.sqrt(1 - abar[t])) < 0.02
    assert abs(step_samples.std() - clsd_samples.std()) < 0.05      # 两种走法方差吻合

    # ② 信噪比随 t 单调下降:t 越大信号越被淹没,这正是模型要逆转的
    print("\nt      √ᾱ(信号占比)   SNR=ᾱ/(1-ᾱ)")
    snrs = []
    for tt in [0, 100, 400, 999]:
        snr = abar[tt] / (1 - abar[tt])
        snrs.append(snr)
        print(f"{tt:4d}   {np.sqrt(abar[tt]):.3f}          {snr:8.3f}")
    assert all(snrs[i] > snrs[i + 1] for i in range(len(snrs) - 1))   # 严格递减

    # ③ 已知真噪声 ε 时可反解 x₀(逆过程的"理想目标":模型学的就是预测这个 ε)
    eps = rng.standard_normal(x0.shape)
    xt = forward_closed(x0, t, eps)
    x0_hat = (xt - np.sqrt(1 - abar[t]) * eps) / np.sqrt(abar[t])
    assert np.allclose(x0_hat, x0, atol=1e-6)

    print("\n✅ 前向加噪机制成立:闭式=逐步链、SNR 随 t 递减、已知 ε 可精确反解 x₀")
    print("面试 Q：为何训练用闭式 x_t=√ᾱ·x₀+√(1-ᾱ)·ε 而不逐步加噪？"
          " A：闭式可对任意 t 一步采样,训练时随机抽 t 直接算,无需真的跑 t 步,极大提速且方差可控。")
