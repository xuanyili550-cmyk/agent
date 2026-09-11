"""
================================================================================
 ML for 3D · Chapter 5 · LGM 图像转 3D（学习笔记 · 高斯参数 numpy 演示 + 🔴真实pipeline参考）
================================================================================
 一句话：LGM(Large Gaussian Model)把"多视图扩散 + 高斯泼溅"打包成 image→3D 端到端,配 Gradio demo。
 本章讲：
   ① 流程：单图 →(多视图扩散,见 ch2)→ 多视图 →(大模型)→ 一堆 3D 高斯(见 ch3 泼溅成像)。
   ② 每个 3D 高斯的参数(本文件 numpy 演示形状)：位置3 + 缩放3 + 旋转四元数4 + 不透明度1 + 颜色3 = 14 维。
   ③ Gradio 包成交互 demo(上传图 → 出 3D)。
 要点：LGM 是当下 image→3D 的代表;输出是"高斯点云参数",可实时渲染。
 跑：python3 chapter5_LGM图像转3D_学习笔记.py   （参数演示纯 numpy 真跑;真实 LGM 见 run_real 🔴需 CUDA+特殊轮子)
================================================================================
"""
import numpy as np


def gaussian_params(n=2048):
    """LGM 输出:n 个 3D 高斯,每个 14 维参数。这里造随机参数看形状/含义。"""
    pos = np.random.randn(n, 3)          # 位置 xyz
    scale = np.abs(np.random.randn(n, 3))  # 缩放(各轴)
    rot = np.random.randn(n, 4)          # 旋转(四元数)
    opacity = np.random.rand(n, 1)       # 不透明度
    color = np.random.rand(n, 3)         # 颜色 rgb
    return np.concatenate([pos, scale, rot, opacity, color], axis=1)


def run_real(image):   # 🔴 需 CUDA + diff_gaussian_rasterization 轮子,默认不调用
    import torch
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained("dylanebert/LGM-full",
                                             custom_pipeline="dylanebert/LGM-full",
                                             torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
    return pipe(image)


def main():
    g = gaussian_params()
    assert g.shape == (2048, 14)         # n 个高斯 × 14 维参数
    print(f"✅ Ch5 跑通：LGM 3D 高斯参数张量 {g.shape}(每点 14 维:位置3+缩放3+旋转4+不透明1+颜色3)")
    print("   流程:单图→多视图扩散(ch2)→LGM→3D 高斯→泼溅渲染(ch3);Gradio 包成 demo。")
    # 面试：Q image→3D 代表方案? A LGM(多视图扩散+大高斯模型); Q 输出是什么? A 3D 高斯点云参数,可实时渲染。


if __name__ == "__main__":
    main()
