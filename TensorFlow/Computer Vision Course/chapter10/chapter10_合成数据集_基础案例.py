"""
 CV Course · Ch10 · 基础案例：程序化合成数据集 = 图像+分类标签+分割掩码全自动(纯 numpy,可跑)
 合成数据的核心优势:标注是"生成时就知道的真值",无需人工标注,还能随机化位置/大小/噪声无限造样本。
 这里从零写一个生成器:随机在画布上画圆或方,同时吐出①图像 ②类别标签 ③像素级掩码(前景真值),
 再加高斯噪声模拟真实成像。验证掩码与图像前景一致、数据集类别均衡。跑：python3 本文件
"""
import numpy as np


def render_shape(rng, size=32):
    """随机渲染一个形状:返回 (图像, 类别标签, 分割掩码)。标签和掩码都是生成时的真值,零人工标注。"""
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = rng.integers(8, size - 8, size=2)           # 随机中心(位置随机化 = 数据增强的一种)
    r = rng.integers(4, 7)                               # 随机半径/半边长
    label = int(rng.integers(0, 2))                      # 0=圆 1=方
    if label == 0:
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2)
    else:
        mask = (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)
    img = mask.astype(np.float32)                        # 前景=1 背景=0(干净的合成图)
    img = np.clip(img + rng.normal(0, 0.1, img.shape), 0, 1)  # 加高斯噪声,模拟真实传感器
    return img, label, mask


def make_dataset(n=200, seed=0):
    """批量造数据集:每个样本都自带三件套。返回图像批、标签批、掩码批。"""
    rng = np.random.default_rng(seed)
    imgs, labels, masks = [], [], []
    for _ in range(n):
        img, label, mask = render_shape(rng)
        imgs.append(img); labels.append(label); masks.append(mask)
    return np.array(imgs), np.array(labels), np.array(masks)


if __name__ == "__main__":
    imgs, labels, masks = make_dataset(200)

    # ① 掩码是真值:去掉噪声后,图像前景应与掩码高度吻合(用阈值恢复前景比对)
    recovered = imgs[0] > 0.5
    iou = (recovered & masks[0]).sum() / ((recovered | masks[0]).sum() + 1e-9)
    assert iou > 0.8                                     # 掩码/图像自洽(合成的标注天然准)

    # ② 类别标签是真值,可直接统计分布,验证数据集大致均衡(无采集偏差)
    n_circle = int((labels == 0).sum()); n_square = int((labels == 1).sum())
    assert abs(n_circle - n_square) < 0.3 * len(labels)  # 两类数量接近

    # ③ 掩码像素数与形状一致:方形前景面积应普遍大于圆(几何真值可交叉核对)
    circle_area = masks[labels == 0].sum(axis=(1, 2)).mean()
    square_area = masks[labels == 1].sum(axis=(1, 2)).mean()
    assert square_area > circle_area

    print(f"数据集: {len(imgs)} 张 {imgs.shape[1]}x{imgs.shape[2]}  圆 {n_circle} 张 / 方 {n_square} 张")
    print(f"样本0 掩码-图像 IoU={iou:.2f}  平均前景面积: 圆≈{circle_area:.0f} 方≈{square_area:.0f} 像素")
    print("✅ 程序化合成:图像+标签+掩码一次生成、标注即真值、可无限随机化,零人工标注成本")
    # 面试 Q&A：合成数据最大风险是什么?——sim-to-real 域差(合成分布与真实分布不一致),
    #          常靠域随机化(狂加噪声/光照/纹理变化)或域自适应缩小差距,否则模型在真实数据上掉点。
