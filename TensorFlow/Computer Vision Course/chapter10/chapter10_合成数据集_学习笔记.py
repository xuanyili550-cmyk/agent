"""
================================================================================
 CV Course · Chapter 10 · 合成数据集（学习笔记 · 程序化造数据 numpy 可跑）
================================================================================
 一句话：真实标注贵又少,用程序/3D 渲染器造"自带完美标注"的合成训练数据。
 本章讲(纯 numpy 程序化生成带标签的形状图,无需 Blender)：
   ① 程序化生成:在网格上画圆/方,自动得到 图像 + 类别标签(+ 分割掩码/边界框)。
   ② PBR 渲染(Blender/Unity):基于物理的光照材质,让合成图逼真。
   ③ sim-to-real:合成数据自带标注,但要缩小与真实的"域差"。
 要点：合成数据自带完美标注(分割/深度/位姿全免费);难点是域差。
 跑：python3 chapter10_合成数据集_学习笔记.py   （程序化造数据纯 numpy 真跑)
================================================================================
"""
import numpy as np


def make_sample(shape="circle", size=32, rng=None):
    """在 size×size 网格上画一个形状 → (图像, 标签, 掩码)。标注是自动的。"""
    rng = rng or np.random.default_rng()
    img = np.zeros((size, size))
    cy, cx = rng.integers(8, size - 8, 2)
    yy, xx = np.mgrid[0:size, 0:size]
    if shape == "circle":
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= 6 ** 2
        label = 0
    else:  # square
        mask = (np.abs(yy - cy) <= 5) & (np.abs(xx - cx) <= 5)
        label = 1
    img[mask] = 1.0
    return img, label, mask


def make_dataset(n=100):
    rng = np.random.default_rng(0)
    data = [make_sample("circle" if i % 2 == 0 else "square", rng=rng) for i in range(n)]
    X = np.array([d[0] for d in data]); y = np.array([d[1] for d in data])
    return X, y


def main():
    X, y = make_dataset(100)
    assert X.shape == (100, 32, 32) and set(y.tolist()) == {0, 1}
    print(f"✅ Ch10 跑通：程序化生成 {len(X)} 张带标签形状图 {X.shape}(圆=0/方=1,标注全自动)")
    print("   真实:Blender/Unity PBR 渲染逼真图 + 免费的分割/深度/位姿标注;注意缩小 sim-to-real 域差。")
    # 面试：Q 合成数据好处? A 标注免费且完美; Q 最大坑? A 域差(合成≠真实),要域随机化/域适应。


if __name__ == "__main__":
    main()
