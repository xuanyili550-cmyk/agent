"""
 Diffusion Course · Ch1 · 基础案例：前向扩散加噪(纯 numpy,可跑)
 看清"随时间步 t 增大,信号被噪声淹没"——扩散模型要学的就是逆转这个过程。跑：python3 本文件
"""
import numpy as np

T = 200
betas = np.linspace(1e-4, 0.02, T)
abar = np.cumprod(1 - betas)                 # 累计保留比例
rng = np.random.default_rng(0)

x0 = np.ones(32) * 0.8                        # 干净信号
for t in [0, 100, 199]:
    eps = rng.standard_normal(x0.shape)
    xt = np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps
    print(f"t={t:3d}  保留比例√ᾱ={np.sqrt(abar[t]):.3f}  x_t 均值={xt.mean():+.2f} std={xt.std():.2f}")
print("✅ t 越大 → 原信号占比越小、越接近标准正态噪声(逆过程=去噪生成)")
