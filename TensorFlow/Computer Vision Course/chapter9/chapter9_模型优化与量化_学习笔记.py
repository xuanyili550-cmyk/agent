"""
================================================================================
 CV Course · Chapter 9 · 模型优化与量化（学习笔记 · int8 量化 numpy 可跑）
================================================================================
 一句话：把训好的模型压小压快好部署——量化(降精度)、剪枝(去冗余)、蒸馏(小学大)。
 本章讲(纯 numpy 手写 float32→int8 量化/反量化,看清怎么压 4 倍)：
   ① 量化:把 float32 权重线性映射到 int8(0~255) → 体积缩 4 倍、算得快。
   ② 反量化:q·scale + zero_point 近似还原,有小误差。
   ③ 剪枝/蒸馏、TF Model Optimization/ONNX 等工具。
 要点：量化用"scale + zero_point"线性映射;误差换体积/速度,通常精度损失很小。
 跑：python3 chapter9_模型优化与量化_学习笔记.py   （量化纯 numpy 真跑)
================================================================================
"""
import numpy as np


def quantize(x, bits=8):
    """对称/仿射量化:float → 整数 + (scale, zero_point)。"""
    qmax = 2 ** bits - 1
    lo, hi = float(x.min()), float(x.max())
    scale = (hi - lo) / qmax
    q = np.round((x - lo) / scale).astype(np.int32)
    return q, scale, lo


def dequantize(q, scale, zero):
    return q * scale + zero


def main():
    rng = np.random.default_rng(0)
    w = rng.standard_normal(1000).astype(np.float32)     # float32 权重
    q, scale, zero = quantize(w)
    w_hat = dequantize(q, scale, zero)
    err = np.abs(w - w_hat).mean()
    assert q.min() >= 0 and q.max() <= 255 and err < 0.01   # int8 范围 + 误差很小
    print(f"✅ Ch9 跑通：float32→int8 量化(体积↓4×),平均反量化误差 {err:.4f}(精度损失极小)")
    # 面试：Q 量化怎么压? A float 线性映射到 int8,scale+zero_point 记录映射; Q 代价? A 少量精度损失换体积/速度。


if __name__ == "__main__":
    main()
