"""
================================================================================
 ML for 3D · Chapter 2 · 生成式 3D 流程（学习笔记 · 多视图投影 numpy 可跑）
================================================================================
 一句话：从"一张图/一句话"生成 3D 的主流路线——先生成"多个视角一致"的图,再重建成 3D。
 本章讲(纯 numpy 演示"多视图"是什么:同一 3D 物体绕轴旋转后投影到 2D,无需模型)：
   ① 多视图扩散：输入单图 → 生成该物体多个角度的图(视角一致是关键)。
   ② 3D 重建：把多视图喂给重建器(高斯泼溅/NeRF/LGM)得到 3D。
   ③ 本文件用一个立方体点集,绕 y 轴旋转 + 正交投影,得到不同视角的 2D 坐标。
 要点：多视图一致性是 image→3D 的桥梁;真实多视图扩散见 gen_real(🔴需 CUDA/diffusers)。
 跑：python3 chapter2_生成式3D流程_学习笔记.py   （投影演示纯 numpy 真跑)
================================================================================
"""
import numpy as np


def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def project(points3d, angle):
    """绕 y 轴旋转后正交投影到 xy 平面 → 一个视角的 2D 坐标。"""
    return (points3d @ rot_y(angle).T)[:, :2]


def gen_real(image_path):   # 🔴 需 CUDA/diffusers,默认不调用
    import torch
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained("dylanebert/multi-view-diffusion",
                                             custom_pipeline="dylanebert/multi-view-diffusion",
                                             torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
    return pipe(image_path)   # → 多视图图像


def main():
    # 立方体 8 个角点
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    v0 = project(cube, 0.0)
    v90 = project(cube, np.pi / 2)
    assert v0.shape == (8, 2) and not np.allclose(v0, v90)   # 不同视角投影不同
    print(f"✅ Ch2 跑通：立方体从 2 个视角投影 → 各 {v0.shape} 的 2D 坐标(视角不同结果不同)")
    print("   真实：单图→多视图扩散(视角一致)→重建 3D;见 ch3 高斯泼溅、ch5 LGM。")
    # 面试：Q image→3D 关键中间步? A 生成视角一致的多视图; Q 再怎么变 3D? A 高斯泼溅/NeRF 重建。


if __name__ == "__main__":
    main()
