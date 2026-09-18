"""
 CV Course · Ch1 · 基础案例：特征描述子匹配全流程(纯 numpy,机制真实,可跑)
 特征匹配三步:① 每个关键点算一个描述子向量 → ② 到另一张图找最近邻 → ③ 用 Lowe 比率+互为最近邻剔除误匹配。
 这里不装 OpenCV,纯 numpy 造两张图的描述子、真算 L2 距离矩阵、真跑比率测试与互检,把"为什么能筛掉误匹配"讲透。
 跑：python3 本文件
"""
import numpy as np

rng = np.random.default_rng(1)


def match_ratio_test(descA, descB, ratio=0.75):
    """Lowe 比率测试:对 A 的每个描述子,看它到 B 的最近邻是否"明显好过"次近邻。
    为什么:纹理重复处会有多个同样近的候选(模糊),此时最近/次近≈1,应丢弃;真匹配才够独特。"""
    matches = []
    for i, a in enumerate(descA):
        dist = np.linalg.norm(descB - a, axis=1)     # A[i] 到 B 所有描述子的欧氏距离
        j1, j2 = np.argsort(dist)[:2]                 # 最近邻 j1、次近邻 j2
        if dist[j1] < ratio * dist[j2]:               # 最近邻显著更近才收(比率<0.75)
            matches.append((i, j1, dist[j1]))
    return matches


def mutual_check(matchesAB, matchesBA):
    """互为最近邻(cross-check):A→B 认 j,同时 B→A 也认回 i,才算稳。单向匹配易受遮挡/重复纹理骗。"""
    back = {b_idx: a_idx for b_idx, a_idx, _ in matchesBA}   # B→A 的最优对应:B下标 → A下标
    return [(a, b, d) for a, b, d in matchesAB if back.get(b) == a]


if __name__ == "__main__":
    # 造 6 个"真实关键点"描述子;B 是 A 加噪+洗牌,再混入 4 个干扰点(无真匹配)
    keypts = rng.standard_normal((6, 8))
    perm = rng.permutation(6)
    descA = keypts
    descB = np.vstack([keypts[perm] + 0.05 * rng.standard_normal((6, 8)),
                       rng.standard_normal((4, 8))])   # 后 4 行是干扰,不该被匹配上

    ab = match_ratio_test(descA, descB)
    ba = match_ratio_test(descB, descA)
    good = mutual_check(ab, ba)

    print(f"比率测试后 A→B 保留 {len(ab)} 对;互检后剩 {len(good)} 对稳定匹配")
    for a, b, d in sorted(good):
        truth = "✅正确" if perm[b] == a else "✗错配"    # descB[b] 是 keypts[perm[b]] 的噪声副本
        print(f"  A[{a}] ↔ B[{b}]  距离{d:.3f}  {truth}")

    assert all(perm[b] == a for a, b, _ in good)       # 留下的必须全是真对应
    assert len(good) >= 5                                # 6 个真点应能召回绝大多数
    print("✅ 比率测试剔模糊 + 互检剔单向误配 → 只留高置信匹配(SIFT/ORB 匹配的通用套路)")
    # 面试Q:Lowe 比率测试为何用"最近/次近"而非"最近邻绝对距离"阈值?
    #      A:绝对阈值对不同图像/尺度不通用;比率是相对判据——真匹配应远优于次优,重复纹理处比率≈1 会被自动淘汰。
