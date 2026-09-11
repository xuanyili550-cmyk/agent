"""
 Diffusion Course · Ch2 · 基础案例：无分类器引导(CFG)公式(纯 numpy,可跑)
 ε_guided = ε_uncond + scale·(ε_cond − ε_uncond)。跑：python3 本文件
"""
import numpy as np

rng = np.random.default_rng(1)
eps_uncond = rng.standard_normal(5)
eps_cond = rng.standard_normal(5)
for scale in [0.0, 1.0, 3.0, 7.5]:
    guided = eps_uncond + scale * (eps_cond - eps_uncond)
    print(f"scale={scale:4}  ‖guided-uncond‖={np.linalg.norm(guided-eps_uncond):.2f}")
print("✅ scale=0→无条件, 1→纯条件, 越大越强引导(偏离无条件越远,可能过饱和)")
