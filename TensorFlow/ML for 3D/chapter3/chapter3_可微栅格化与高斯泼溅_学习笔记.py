"""
================================================================================
 ML for 3D · Chapter 3 · 可微栅格化与高斯泼溅（学习笔记 · splat numpy 可跑）
================================================================================
 一句话：把"3D→2D 渲染"做成可微的,梯度就能从图像 loss 回传 → 用 2D 图监督优化 3D 表示。
 本章讲(纯 numpy 实现极简高斯泼溅:投影+按深度排序+累加成像,无需 CUDA)：
   ① 高斯泼溅(3DGS)：场景=一堆带位置/颜色/不透明度的 3D 高斯,投影到 2D 后"泼"成像素。
   ② 深度排序(画家算法)：远的先画、近的后画(近处覆盖远处)。
   ③ 可微：整个投影+混合可求导 → 能端到端优化(用多视图 loss 拟合场景)。
 要点：可微渲染 = 用 2D 图像 loss 反向优化 3D 参数;比 NeRF 快、可实时。
 跑：python3 chapter3_可微栅格化与高斯泼溅_学习笔记.py   （splat 纯 numpy 真跑)
================================================================================
"""
import numpy as np


def splat(points3d, colors, size=16):
    """极简高斯泼溅:3D 点正交投影到 size×size 图,按深度从远到近累加颜色。"""
    img = np.zeros((size, size, 3))
    order = np.argsort(-points3d[:, 2])            # z 大(远)的先画
    for i in order:
        px = int((points3d[i, 0] * 0.5 + 0.5) * (size - 1))
        py = int((points3d[i, 1] * 0.5 + 0.5) * (size - 1))
        px, py = np.clip([px, py], 0, size - 1)
        img[py, px] = np.clip(img[py, px] + colors[i], 0, 1)   # "泼"上颜色
    return img


def main():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (30, 3))              # 30 个 3D 高斯点
    cols = rng.uniform(0, 1, (30, 3))
    img = splat(pts, cols)
    assert img.shape == (16, 16, 3) and img.sum() > 0
    print(f"✅ Ch3 跑通：30 个 3D 高斯 → 泼溅成 {img.shape} 图(非空像素 {int((img.sum(-1)>0).sum())} 个)")
    print("   可微版:投影+混合可求导 → 用多视图图像 loss 端到端优化这些高斯参数。")
    # 面试：Q 高斯泼溅 vs NeRF? A 显式高斯点、更快可实时; Q 为什么要可微? A 用 2D 图 loss 优化 3D。


if __name__ == "__main__":
    main()
