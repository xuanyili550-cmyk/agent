"""
================================================================================
 Audio Course · Chapter 3 · 音频的 Transformer 架构（学习笔记 · 可运行真代码）
================================================================================
 一句话：Transformer 靠"注意力"建模序列；音频任务按输入/输出形态分成 ASR/分类/TTS/S2S。
 本章讲(用纯 numpy 手写注意力，看清它到底算什么，无需下模型)：
   ① 自注意力：softmax(QKᵀ/√d)·V —— 每个位置按相关性加权聚合其它位置的信息。
   ② 编码器(理解输入) / 解码器(自回归生成) / 交叉注意力(解码器看编码器)。
   ③ 音频任务形态：ASR(语音→文本,seq2seq) / 分类(语音→标签,仅编码器) / TTS(文本→语音) / S2S。
 要点：选架构看任务——要"理解"用编码器、要"生成"用解码器、"一进一出"用编解码。
 跑：python3 chapter3_音频Transformer架构_学习笔记.py   （纯 numpy，无需联网）
================================================================================
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)     # 数值稳定：减最大值防上溢
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ① 自注意力(单头)：Q,K,V 形如 [seq, d]
def self_attention(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)               # 相关性打分 [seq, seq]
    weights = softmax(scores, axis=-1)          # 每行归一化=对各位置的关注度
    out = weights @ V                           # 按关注度加权聚合 V
    return out, weights


def main():
    rng = np.random.default_rng(0)
    seq, d = 5, 8                               # 5 个时间步(如 5 帧音频特征)，每步 8 维
    X = rng.standard_normal((seq, d))
    # 演示：Q=K=V=X 的自注意力(实际会各自过一个线性层)
    out, w = self_attention(X, X, X)
    assert out.shape == (seq, d)
    assert np.allclose(w.sum(axis=-1), 1.0)     # 每个位置的注意力权重和为 1
    print(f"✅ Ch3 跑通：自注意力 输出{out.shape}，注意力矩阵{w.shape}(每行和=1)")
    print("   架构选型：分类→仅编码器 / 生成→仅解码器 / ASR·翻译→编码器-解码器(交叉注意力)")
    # 面试：Q 除以 √d 干嘛? A 防点积过大使 softmax 饱和梯度消失; Q 交叉注意力? A 解码器 Q 看编码器的 K/V。


if __name__ == "__main__":
    main()
