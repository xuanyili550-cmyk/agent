"""
 ML for 3D · Ch5 · 基础案例：3D 高斯参数张量形状(纯 numpy,可跑)
 LGM 的输出就是"一堆 3D 高斯",每个 14 维。跑：python3 本文件
"""
import numpy as np

n = 1024
gaussians = {
    "position": np.random.randn(n, 3), "scale": np.abs(np.random.randn(n, 3)),
    "rotation": np.random.randn(n, 4), "opacity": np.random.rand(n, 1), "color": np.random.rand(n, 3),
}
total = sum(v.shape[1] for v in gaussians.values())
print("每个 3D 高斯维度:", {k: v.shape[1] for k, v in gaussians.items()}, "合计", total)
assert total == 14
print(f"✅ {n} 个 3D 高斯 → 参数张量 ({n}, {total});可实时泼溅渲染成图")
