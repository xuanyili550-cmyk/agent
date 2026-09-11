"""
 CV Course · Ch5 · 基础案例：VAE 重参数技巧(纯 numpy,可跑)
 z = μ + σ·ε,让"采样"可训练。跑：python3 本文件
"""
import numpy as np

rng = np.random.default_rng(0)
mu, logvar = np.array([1.0, -1.0]), np.array([0.0, 0.0])
std = np.exp(0.5 * logvar)
z = mu + std * rng.standard_normal(2)
print(f"μ={mu}, σ={std}, 采样 z={z.round(2)}")
print("✅ 随机性在 ε,μ/σ 可求导 → 采样这一步能反向传播(VAE 能训的关键)")
