"""
 CV Course · Ch2 · 基础案例：手写一个 CNN 前向块(纯 numpy,机制真实,可跑)
 CNN 的一层 = 多核卷积(提局部特征) → ReLU(非线性,丢负响应) → 最大池化(降采样+平移不变)。
 这里纯 numpy 把这三件套连起来跑一张小图,真算每个数,讲清"为什么"每步都不可少。
 跑：python3 本文件
"""
import numpy as np


def conv2d(img, kernels):
    """多核卷积:每个核在图上滑窗做"逐元素乘再求和",输出一张响应图(feature map)。
    K 个核 → K 张响应图,分别对边缘/角点等不同局部模式敏感 → 这就是 CNN 学"特征"的方式。"""
    H, W = img.shape
    k = kernels.shape[1]                                  # 核边长
    oh, ow = H - k + 1, W - k + 1                          # valid 卷积输出尺寸
    out = np.zeros((len(kernels), oh, ow))
    for c, ker in enumerate(kernels):
        for i in range(oh):
            for j in range(ow):
                out[c, i, j] = (img[i:i+k, j:j+k] * ker).sum()
    return out


def relu(x):
    """非线性:负响应清零。为什么必须有——没有它,多层卷积叠起来仍等价于一层线性,学不到复杂模式。"""
    return np.maximum(0, x)


def maxpool(x, size=2):
    """2×2 最大池化:每块取最大值。降采样省算力,且"最强响应"对小位移不敏感 → 平移不变性。"""
    C, H, W = x.shape
    oh, ow = H // size, W // size
    out = np.zeros((C, oh, ow))
    for c in range(C):
        for i in range(oh):
            for j in range(ow):
                out[c, i, j] = x[c, i*size:i*size+size, j*size:j*size+size].max()
    return out


if __name__ == "__main__":
    img = np.zeros((6, 6)); img[:, 3:] = 1                 # 中间一条竖边
    kernels = np.array([
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],              # 核0:竖直边缘
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],              # 核1:水平边缘
    ])
    feat = conv2d(img, kernels)
    activated = relu(feat)
    pooled = maxpool(activated)

    print("卷积输出 shape:", feat.shape, "→ 池化后:", pooled.shape)
    print("竖边核在边界列响应最大(取绝对值):", np.abs(feat[0]).max().round(1))
    print("水平边核对竖边几乎无响应:", np.abs(feat[1]).max().round(1))
    assert np.abs(feat[0]).max() > np.abs(feat[1]).max()   # 竖边核应比水平核响应强
    assert (activated >= 0).all()                          # ReLU 后无负值
    assert pooled.shape == (2, 2, 2)                       # 4×4 → 池化 → 2×2
    print("✅ 卷积(提特征)+ReLU(非线性)+池化(降采样&平移不变)三件套 = CNN 一层的完整前向")
    # 面试Q:池化和步长卷积都能降采样,为什么早期 CNN 偏爱最大池化?
    #      A:最大池化取局部最强响应,天然带平移不变性且无参数;现代网络也常用步长卷积让降采样可学习,各有取舍。
