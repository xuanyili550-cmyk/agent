"""
================================================================================
 CV Course · Chapter 4 · 多模态世界与 CLIP 对齐（学习笔记 · 图文匹配 numpy 可跑）
================================================================================
 一句话：真实信息是多模态的;跨模态关键在"把图和文对齐到同一向量空间"(CLIP 思路),就能互相检索/零样本分类。
 本章讲(纯 numpy 演示 CLIP 式图文匹配:归一化+余弦,无需模型)：
   ① 多模态任务:VQA/图像描述/OCR/文生图/文生视频。
   ② CLIP:图编码器+文编码器把图/文各编成向量,训练让"配对的图文"余弦相似度高。
   ③ 用途:给一张图,从多个候选描述里挑最匹配的(零样本分类同理)。
 要点：对齐到共享空间后,余弦相似度=跨模态可比;归一化后点积=余弦。
 跑：python3 chapter4_多模态与CLIP对齐_学习笔记.py   （图文匹配纯 numpy 真跑)
================================================================================
"""
import numpy as np


def l2norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def clip_match(image_emb, text_embs, captions):
    """图向量 vs 多个文本向量,余弦最高的就是最匹配描述(CLIP 零样本分类核心)。"""
    sims = l2norm(text_embs) @ l2norm(image_emb)
    best = int(np.argmax(sims))
    return captions[best], sims


def main():
    rng = np.random.default_rng(0)
    img = rng.standard_normal(16)
    # 造 3 个候选描述向量,让第 2 个和图最接近
    texts = rng.standard_normal((3, 16)); texts[1] = img + rng.standard_normal(16) * 0.05
    caps = ["一只狗", "一只猫在沙发上", "一辆车"]
    best, sims = clip_match(img, texts, caps)
    assert best == "一只猫在沙发上"
    print(f"✅ Ch4 跑通：CLIP 式图文匹配 → 最匹配'{best}'(余弦 {sims.round(2)})")
    # 面试：Q CLIP 怎么零样本分类? A 图向量和各类别文本向量比余弦,取最高; Q 对齐靠什么? A 对比学习拉近配对图文。


if __name__ == "__main__":
    main()
