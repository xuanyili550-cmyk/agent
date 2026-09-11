"""
================================================================================
 CV Course · Chapter 2 · 卷积神经网络(CNN)（学习笔记 · conv/pool numpy 可跑）
================================================================================
 一句话：CV 的地基——卷积提局部特征、池化降维,堆叠成深网(LeNet→VGG→ResNet)。
 本章讲(纯 numpy 手写 conv2d + relu + maxpool,看清卷积到底算什么)：
   ① 卷积：卷积核在图上滑动,每处做"逐元素乘再求和" → 提取边缘/纹理等局部特征。
   ② 激活 ReLU：max(0,x) 引入非线性。
   ③ 池化 MaxPool：每小块取最大 → 降维 + 平移不变。
 要点：CNN 靠"局部感受野 + 权重共享"高效抓图像特征;真实网络见 build_vgg(🔴需 torch)。
 跑：python3 chapter2_卷积神经网络_学习笔记.py   （conv/pool 纯 numpy 真跑)
================================================================================
"""
import numpy as np


def conv2d(img, kernel):
    kh, kw = kernel.shape
    H, W = img.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = (img[i:i + kh, j:j + kw] * kernel).sum()   # 逐元素乘求和
    return out


def relu(x):
    return np.maximum(0, x)


def maxpool(x, k=2):
    H, W = x.shape
    out = np.zeros((H // k, W // k))
    for i in range(0, H - H % k, k):
        for j in range(0, W - W % k, k):
            out[i // k, j // k] = x[i:i + k, j:j + k].max()
    return out


def build_vgg():   # 🔴 需 torch,默认不调用
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU(), nn.MaxPool2d(2))


def main():
    img = np.zeros((6, 6)); img[:, 3:] = 1.0                 # 左黑右白,中间一条竖边
    edge_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # 竖直边缘检测
    feat = relu(conv2d(img, edge_kernel))
    pooled = maxpool(feat)
    assert feat.max() > 0 and pooled.shape == (2, 2)
    print(f"✅ Ch2 跑通：卷积检测到竖边(响应最大 {feat.max():.0f}) → ReLU → MaxPool 到 {pooled.shape}")
    # 面试：Q 卷积为何适合图像? A 局部感受野+权重共享,参数少且平移不变; Q 池化作用? A 降维+平移不变。


if __name__ == "__main__":
    main()
