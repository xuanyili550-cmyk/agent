"""
 CV Course · Ch9 · 基础案例：int8 量化往返 + 误差分析(对称/非对称,纯 numpy,可跑)
 量化 = 把 float32 权重映射到 8bit 整数(体积缩 4 倍、整数乘加更快),推理时再反量化回近似 float。
 关键是 scale/zero-point 怎么定:非对称(用 min~max 全量程)vs 对称(用 ±max、zero-point=0)。
 这里两种都从零实现完整"量化→存整数→反量化"往返,真算量化误差,并验证误差有理论上界(≤ scale/2)。
 跑：python3 本文件
"""
import numpy as np


def quantize_asymmetric(w, bits=8):
    """非对称量化:用真实 min~max 撑满 [0, 2^bits-1],需要 zero_point 记录 0 落在哪个整数。"""
    qmin, qmax = 0, 2 ** bits - 1
    lo, hi = float(w.min()), float(w.max())
    scale = (hi - lo) / (qmax - qmin)                    # 每个整数刻度代表多少 float
    zero_point = np.round(qmin - lo / scale)             # float 的 0 对应的整数
    q = np.clip(np.round(w / scale + zero_point), qmin, qmax).astype(np.int32)
    return q, scale, zero_point


def dequantize_asymmetric(q, scale, zero_point):
    """反量化:整数 → 近似 float。真实推理里权重就以 q 存储,用时才还原。"""
    return (q.astype(np.float32) - zero_point) * scale


def quantize_symmetric(w, bits=8):
    """对称量化:zero_point 恒为 0,量程用 [-max, +max],映射到 [-127,127](部署更常用,乘加省一步)。"""
    qmax = 2 ** (bits - 1) - 1                            # 127
    scale = float(np.abs(w).max()) / qmax
    q = np.clip(np.round(w / scale), -qmax, qmax).astype(np.int32)
    return q, scale


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    w = rng.standard_normal(1000).astype(np.float32)     # 一层权重(近似正态)

    q_a, scale_a, zp = quantize_asymmetric(w)
    w_hat_a = dequantize_asymmetric(q_a, scale_a, zp)
    q_s, scale_s = quantize_symmetric(w)
    w_hat_s = q_s.astype(np.float32) * scale_s

    err_a = np.abs(w - w_hat_a).max()
    err_s = np.abs(w - w_hat_s).max()
    # 量化本质是"四舍五入到最近刻度",单点误差理论上 ≤ scale/2(加一点 clip/浮点余量)
    assert err_a <= scale_a / 2 + 1e-5
    assert 0 <= q_a.min() and q_a.max() <= 255           # 非对称落在 uint8 区间
    assert -127 <= q_s.min() and q_s.max() <= 127        # 对称落在 int8 区间
    assert w.nbytes == 4 * q_a.astype(np.int8).nbytes    # float32 4 字节 → int8 1 字节,缩 4 倍

    print(f"非对称: scale={scale_a:.4f} zp={zp:.0f}  最大误差={err_a:.4f}  (理论上界 scale/2={scale_a/2:.4f})")
    print(f"对称  : scale={scale_s:.4f} zp=0        最大误差={err_s:.4f}")
    print(f"体积: float32={w.nbytes}B → int8={q_a.astype(np.int8).nbytes}B  缩 {w.nbytes // q_a.astype(np.int8).nbytes} 倍")
    print("✅ int8 量化往返跑通:两种方案误差都受 scale/2 约束,体积缩 4 倍")
    # 面试 Q&A：对称 vs 非对称怎么选?——权重多近似零对称,用对称(zp=0 免去乘加里的偏置修正、更快);
    #          激活常单边分布(如 ReLU 后全≥0),用非对称能撑满量程、误差更小。
