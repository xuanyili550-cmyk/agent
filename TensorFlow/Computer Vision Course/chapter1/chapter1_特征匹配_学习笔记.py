"""
================================================================================
 CV Course · Chapter 1 · 特征匹配（学习笔记 · 描述子匹配 numpy 可跑）
================================================================================
 一句话：在两张图里找"同一个点"——检测关键点、算描述子(向量)、再按距离匹配 + 比率测试筛掉误匹配。
 本章讲(纯 numpy 演示"描述子匹配"核心,无需 opencv)：
   ① 描述子：每个关键点用一个向量表示其局部外观(SIFT/ORB)。
   ② 最近邻匹配：A 的每个描述子找 B 里最近的;knn 取最近两个。
   ③ Lowe 比率测试：最近/次近 < 0.75 才算好匹配(过滤模糊匹配)。
 要点：比率测试是去误匹配的关键;用于拼接/配准/SLAM/检索。真实 SIFT 见 match_real(🔴需 opencv)。
 跑：python3 chapter1_特征匹配_学习笔记.py   （匹配核心纯 numpy 真跑)
================================================================================
"""
import numpy as np


def knn_match(desA, desB, ratio=0.75):
    """A 每个描述子在 B 找最近两个,过 Lowe 比率测试的才保留。返回 [(i,j)]。"""
    good = []
    for i, a in enumerate(desA):
        d = np.linalg.norm(desB - a, axis=1)     # 到 B 所有描述子的距离
        j1, j2 = np.argsort(d)[:2]               # 最近、次近
        if d[j1] < ratio * d[j2]:                # 比率测试
            good.append((i, int(j1)))
    return good


def match_real(img1, img2):   # 🔴 需 opencv-python,默认不调用
    import cv2
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    return cv2.BFMatcher().knnMatch(d1, d2, k=2)


def main():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((5, 8))                 # 5 个描述子
    desB = np.vstack([base, rng.standard_normal((5, 8))])  # B 含这 5 个 + 5 个干扰
    desA = base + rng.standard_normal((5, 8)) * 0.01   # A 是 base 的轻微扰动
    good = knn_match(desA, desB)
    assert len(good) == 5 and all(i == j for i, j in good)  # 应精确匹配到前 5 个
    print(f"✅ Ch1 跑通：描述子匹配 + 比率测试 → {len(good)} 对好匹配 {good}")
    # 面试：Q 比率测试干嘛? A 最近/次近<0.75 过滤模糊匹配; Q 描述子? A 关键点局部外观的向量表示。


if __name__ == "__main__":
    main()
