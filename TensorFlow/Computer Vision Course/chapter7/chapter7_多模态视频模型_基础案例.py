"""
 CV Course · Ch7 · 基础案例：视频时序建模——mean 池化 vs 注意力池化(纯 numpy,机制真实,可跑)
 视频 = T 帧,每帧一个特征向量;要聚合成 1 个视频级向量喂下游。最简单是 mean(所有帧等权)。
 但关键动作往往只在少数帧;注意力池化学一个"帧重要性权重",让信息量大的帧占更大比重 → 更贴近真实视频模型。
 这里纯 numpy 真算两种池化,并造一个"只有第 3 帧有信号、其余是噪声"的视频,验证注意力能聚焦到关键帧。
 跑：python3 本文件
"""
import numpy as np


def mean_pool(video):
    """时序 mean 池化:T 帧等权平均。简单但关键帧会被大量背景帧稀释。"""
    return video.mean(0)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def attention_pool(video, query):
    """注意力池化:每帧与一个可学习 query 算相似度打分 → softmax 成权重 → 加权求和。
    关键帧(与 query 更像)拿到更高权重,不会被噪声帧淹没。真实视频 Transformer 就是这套 [CLS]/query 聚合。"""
    scores = video @ query / np.sqrt(video.shape[1])       # 每帧一个分数
    weights = softmax(scores)
    return weights @ video, weights


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, d = 5, 8
    signal = np.ones(d)                                     # "关键动作"的特征方向
    video = 0.2 * rng.standard_normal((T, d))              # 背景噪声帧
    video[2] += 2.0 * signal                                # 只在第 3 帧(idx=2)注入强信号

    query = signal / np.linalg.norm(signal)                 # query 指向我们关心的动作方向
    mean_vec = mean_pool(video)
    attn_vec, weights = attention_pool(video, query)

    print("逐帧特征 shape:", video.shape)
    print("注意力权重(应在第 3 帧最高):", weights.round(3), "→ 关键帧 idx", int(weights.argmax()))
    print("mean 池化与信号方向的相关:", round(float(mean_vec @ query), 3))
    print("注意力池化与信号方向的相关:", round(float(attn_vec @ query), 3))
    assert weights.argmax() == 2                            # 注意力确实聚焦到含信号的关键帧
    assert weights.sum().round(5) == 1.0                    # 权重是合法分布
    assert (attn_vec @ query) > (mean_vec @ query)          # 注意力保留了更多关键信息,没被噪声稀释
    print("✅ mean 池化等权易稀释关键帧;注意力池化按重要性加权 → 视频级表示更聚焦(视频 Transformer 的聚合内核)")
    # 面试Q:视频建模里,时序注意力/池化相比 3D 卷积有什么优势?
    #      A:注意力能建长程依赖、动态聚焦关键帧且权重可解释;3D 卷积感受野受核大小限制、算力随时长快速膨胀,故长视频多用注意力聚合。
