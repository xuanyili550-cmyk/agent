"""
 Diffusion Course · Ch4 · 基础案例：DDIM 由 x_t 反解 x₀(纯 numpy,可跑)
 x₀̂ = (x_t − √(1-ᾱ_t)·ε)/√ᾱ_t —— DDIM 确定性的核心,可逆(反演)的基础。跑：python3 本文件
"""
import numpy as np

rng = np.random.default_rng(2)
x0 = rng.standard_normal(6)
abar_t = 0.25
eps = rng.standard_normal(6)
x_t = np.sqrt(abar_t) * x0 + np.sqrt(1 - abar_t) * eps      # 前向到某步
x0_hat = (x_t - np.sqrt(1 - abar_t) * eps) / np.sqrt(abar_t)  # DDIM 反解
assert np.allclose(x0_hat, x0, atol=1e-6)
print("✅ 已知 ε 时,DDIM 能从 x_t 精确反解 x₀ → 确定性、可逆(这是反演编辑的前提)")
