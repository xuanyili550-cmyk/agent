"""
 ML for 3D · Ch5 · 基础案例：LGM 输出的 3D 高斯参数(14 通道)拆解 + 激活 + 泼溅渲染(纯 numpy,机制真实,可跑)
 LGM(Large Gaussian Model)吃多视图图像,直接回归出一大堆 3D 高斯;每个高斯 14 维:
   位置xyz(3) + 尺度scale(3) + 旋转四元数(4) + 不透明度opacity(1) + 颜色rgb(3) = 14。
 网络原始输出是"未激活 logits",要经不同激活映射到物理合法域(σ>0、α∈[0,1]、四元数单位化)。
 这里从零做:形状拆解 → 激活 → 四元数归一化 → 把高斯泼溅成一张图,验证参数是活的。
 跑：python3 本文件
"""
import numpy as np

CHANNELS = {"position": 3, "scale": 3, "rotation": 4, "opacity": 1, "color": 3}   # 合计 14


def split_and_activate(raw):
    """把网络的 (N,14) 原始输出按通道切开,并各自套上正确激活函数映射到合法物理域。"""
    off, out = 0, {}
    for name, c in CHANNELS.items():
        out[name], off = raw[:, off:off + c], off + c
    out["scale"] = np.exp(out["scale"])                          # 尺度必须 >0 → exp 激活
    out["opacity"] = 1 / (1 + np.exp(-out["opacity"]))           # 不透明度∈(0,1) → sigmoid
    out["color"] = 1 / (1 + np.exp(-out["color"]))               # 颜色∈(0,1) → sigmoid
    q = out["rotation"]
    out["rotation"] = q / np.linalg.norm(q, axis=1, keepdims=True)  # 旋转须为单位四元数 → 归一化
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 1024
    raw = rng.standard_normal((n, sum(CHANNELS.values())))       # 模拟 LGM 回归出的原始 logits
    assert raw.shape == (n, 14)
    g = split_and_activate(raw)

    print("每个 3D 高斯的通道维度:", {k: v.shape[1] for k, v in g.items()}, "合计", sum(v.shape[1] for v in g.values()))
    assert sum(v.shape[1] for v in g.values()) == 14
    # 激活后必须落在物理合法域(否则渲染会出 NaN/负尺度/透明度爆表)
    assert (g["scale"] > 0).all(), "尺度经 exp 后应恒正"
    assert (g["opacity"] >= 0).all() and (g["opacity"] <= 1).all(), "不透明度应∈[0,1]"
    assert np.allclose(np.linalg.norm(g["rotation"], axis=1), 1.0), "四元数应为单位长度"

    # 拿前若干个高斯泼溅到一张小图,证明这些参数真能被渲染(近大远小、α 混合)
    SIZE = 20
    img, abuf = np.zeros((SIZE, SIZE, 3)), np.zeros((SIZE, SIZE))
    pos = np.tanh(g["position"][:64])                            # 压到 [-1,1] 便于映射到画布
    for i in np.argsort(-pos[:, 2]):                            # 画家算法:按 z 远→近
        cx = (pos[i, 0] * 0.5 + 0.5) * (SIZE - 1)
        cy = (pos[i, 1] * 0.5 + 0.5) * (SIZE - 1)
        ys, xs = np.mgrid[0:SIZE, 0:SIZE]
        w = g["opacity"][i, 0] * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 2.5 ** 2))
        a = w[..., None]
        img[:] = g["color"][i] * a + img * (1 - a)             # α over 合成
        abuf[:] = w + abuf * (1 - w)
    print(f"用前 64 个高斯泼溅出 {SIZE}×{SIZE} 图,非空像素:{int((abuf > 0.01).sum())}")
    assert (abuf > 0.01).any() and abuf.max() <= 1.0 + 1e-9
    print(f"✅ LGM 输出 ({n},14) 高斯:14=3+3+4+1+3;各通道经 exp/sigmoid/四元数归一化到合法域后可直接实时泼溅成图")
    # 面试 Q&A：为什么网络不直接输出物理量,而要输出 logits 再激活?
    #   A：让网络在无约束实数空间自由回归、梯度稳定;激活函数(exp/sigmoid/归一化)负责把输出
    #      "夹"进合法域(σ>0、α∈[0,1]、单位四元数),既保证物理有效,又保持处处可导可反传。
