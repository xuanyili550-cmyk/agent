"""
 ML for 3D · Ch3 · 基础案例：可微栅格化 · 高斯泼溅(画家算法+α混合,纯 numpy,机制真实,可跑)
 3D 高斯泼溅渲染 = 把每个高斯投影成屏幕上一团"椭圆软斑"(不是硬点!),按深度从远到近排序,
 用 α 混合(over 合成)一层层盖上去。软斑用 2D 高斯衰减 exp(-r²/2σ²) 生成 → 处处可导 → 可微、可反传优化。
 这里从零实现:泼溅核 + 画家算法排序 + α over 合成,不调任何渲染库。
 跑：python3 本文件
"""
import numpy as np

SIZE = 24                                                # 画布分辨率


def splat(canvas, alpha_buf, cx, cy, color, opacity, sigma):
    """把一个高斯泼成软斑,用 over 合成叠到画布上(α 混合,可导)。"""
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    r2 = (xs - cx) ** 2 + (ys - cy) ** 2                 # 每个像素到高斯中心的距离平方
    weight = opacity * np.exp(-r2 / (2 * sigma ** 2))    # 2D 高斯衰减:中心浓、边缘淡(软斑核心)
    a = weight[..., None]                                # (H,W,1) 该高斯在各像素的 α
    # over 合成:new = src*a + dst*(1-a);远的先铺、近的后盖,alpha_buf 记录累计覆盖
    canvas[:] = color * a + canvas * (1 - a)
    alpha_buf[:] = a[..., 0] + alpha_buf * (1 - a[..., 0])
    return canvas, alpha_buf


if __name__ == "__main__":
    # 3 个 3D 高斯:xyz(z 是深度) + 颜色 + 不透明度 + 尺度σ。位置用 [-1,1] 归一化坐标
    gaussians = [
        {"pos": [-0.3, -0.3, 0.9], "color": [1, 0, 0], "opacity": 0.9, "sigma": 4.0},  # 红,最远
        {"pos": [0.3, 0.3, 0.5], "color": [0, 1, 0], "opacity": 0.9, "sigma": 4.0},   # 绿,居中
        {"pos": [0.0, 0.0, 0.1], "color": [0, 0, 1], "opacity": 0.8, "sigma": 3.0},   # 蓝,最近
    ]
    order = sorted(range(len(gaussians)), key=lambda i: -gaussians[i]["pos"][2])       # 画家算法:远→近
    print("深度排序(远→近)后的泼溅顺序:", [["红", "绿", "蓝"][i] for i in order])

    canvas = np.zeros((SIZE, SIZE, 3))
    alpha_buf = np.zeros((SIZE, SIZE))
    for i in order:
        g = gaussians[i]
        cx = (g["pos"][0] * 0.5 + 0.5) * (SIZE - 1)      # 归一化坐标 → 像素坐标
        cy = (g["pos"][1] * 0.5 + 0.5) * (SIZE - 1)
        splat(canvas, alpha_buf, cx, cy, np.array(g["color"], float), g["opacity"], g["sigma"])

    covered = int((alpha_buf > 0.01).sum())
    print(f"非空(被泼到)像素数:{covered} / {SIZE * SIZE}")
    assert covered > 3                                   # 是"软斑"而非单像素硬点 → 覆盖大片
    # 中心像素应偏蓝(最近的蓝高斯最后合成、盖在最上层)
    center = canvas[SIZE // 2, SIZE // 2]
    assert center.argmax() == 2, center.round(2)
    assert alpha_buf.max() <= 1.0 + 1e-9                 # α 混合后不透明度不会超过 1
    print("✅ 高斯泼溅=软斑核+画家算法排序+α over 合成;exp 衰减处处可导,故可微栅格化能对高斯参数反传优化")
    # 面试 Q&A：3D 高斯泼溅为什么能实时又能被优化(相比 NeRF)?
    #   A：它用显式高斯基元 + 光栅化泼溅(前向快,不用逐像素体渲染射线积分);
    #      而泼溅核是连续可导的 exp 衰减,渲染全程可微,可对位置/尺度/颜色/不透明度直接梯度下降拟合多视图。
