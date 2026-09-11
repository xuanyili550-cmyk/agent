"""
================================================================================
 CV Course · Chapter 7 · 多模态视频模型（学习笔记 · 时序池化 numpy 可跑）
================================================================================
 一句话：视频 = 图像序列 + 声音/文本/动作;模型要同时建模"时间维 + 多模态"。
 本章讲(纯 numpy 演示 帧特征的时序聚合,无需模型)：
   ① 视频表示:每帧过图像编码器得一个特征向量 → 一段视频 = 特征序列 [T, d]。
   ② 时序聚合:mean/max 池化,或时序注意力,把 T 帧压成一个视频级向量。
   ③ 融合声音/字幕/动作等模态 → 多模态视频理解/检索/问答。
 要点：比图像多了"时间维",算力/数据成本更高;最简聚合是时序 mean 池化。
 跑：python3 chapter7_多模态视频模型_学习笔记.py   （时序池化纯 numpy 真跑)
================================================================================
"""
import numpy as np


def temporal_pool(frame_feats, method="mean"):
    """帧特征序列 [T,d] → 视频级向量 [d]。"""
    if method == "max":
        return frame_feats.max(axis=0)
    return frame_feats.mean(axis=0)


def temporal_attention_pool(frame_feats):
    """简单时序注意力:按每帧范数当权重加权(重要帧权重大)。"""
    w = np.linalg.norm(frame_feats, axis=1)
    w = w / w.sum()
    return (frame_feats * w[:, None]).sum(axis=0)


def main():
    rng = np.random.default_rng(0)
    video = rng.standard_normal((8, 16))          # 8 帧,每帧 16 维特征
    v_mean = temporal_pool(video)
    v_attn = temporal_attention_pool(video)
    assert v_mean.shape == (16,) and v_attn.shape == (16,)
    print(f"✅ Ch7 跑通：8 帧特征[8,16] → 时序池化 → 视频向量[16](mean 与注意力两种聚合)")
    # 面试：Q 视频比图像难在哪? A 多了时间维,算力/数据更大; Q 怎么聚合帧? A 时序 mean/max 或时序注意力。


if __name__ == "__main__":
    main()
