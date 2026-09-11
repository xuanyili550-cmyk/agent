"""
================================================================================
 分章项目 · Ch5 · 语义搜索引擎（贴 HF Ch5：Datasets 思路 + 嵌入 + 向量检索）
================================================================================
 HF 课程 Ch5 讲“Datasets 库 + 用 FAISS 做语义搜索”，用可运行代码复现其核心：
   ① 数据集视角：把语料当成一张表(Dataset)，给每行【批量算嵌入】作为新列(生产处理大数据的姿势)。
   ② 嵌入：mean 池化(mask 加权，避开 PAD) + 归一化(归一化后点积=余弦)——RAG 检索的地基。
   ③ 向量检索：按余弦相似度找 top-k(FAISS 就是把这步做成能扛百万级的近似最近邻索引)。
   这是 RAG 检索的地基。完整章节材料见 ../../../Chapter 5/；中文版见 ../中文版/中文_语义检索RAG.py；生产向量库见 ../生产架构/生产02。

 ⚠ 本机说明：HF Ch5 原版用 datasets 的 ds.map() 算嵌入、ds.add_faiss_index() 建索引；但本机
   Python3.14 下 datasets 的 fingerprint/pickle 彻底坏了(连 Dataset.from_dict 都崩，见项目环境说明)、
   faiss 也没装，所以这里用【等价的纯 list + numpy 手写版】，逻辑与结果和 map+FAISS 一致。
 跑：python3 Ch5_语义搜索引擎.py
================================================================================
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

CKPT = "sentence-transformers/all-MiniLM-L6-v2"
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModel.from_pretrained(CKPT).to(DEV).eval()

CORPUS = [
    "FAISS builds an index to quickly find similar embeddings.",
    "Semantic search finds documents by meaning, not exact keywords.",
    "Padding makes all sequences in a batch the same length.",
    "Streaming lets you use a dataset larger than memory.",
    "The Trainer API handles the training loop for you.",
    "Dynamic padding pads each batch to its own longest sequence.",
]


def embed(texts, batch_size=8):
    """[②] 批量算嵌入：mean 池化(mask 加权) + 归一化。分批喂入=map(batched=True) 的等价手写版。"""
    vecs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True, return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
        vecs.append(F.normalize(v, p=2, dim=1).cpu().numpy())
    return np.concatenate(vecs, 0)


# ==============================================================================
# ① 数据集视角：把语料+嵌入当成一张表(text 列 + 向量矩阵)
# ==============================================================================
def build_index():
    # print("=" * 70, "\n① 数据集视角：给每行批量算嵌入，建成向量矩阵\n" + "=" * 70)
    matrix = embed(CORPUS)                                  # 批量算(等价 ds.map(batched=True) 的手写版)
    print(f"  语料 {len(CORPUS)} 行  向量矩阵形状={matrix.shape}(行数×向量维度)")
    return matrix


# ==============================================================================
# ②③ 向量检索：按余弦找 top-k(FAISS 索引做的就是这件事的加速版)
# ==============================================================================
def search(matrix, queries, k=2):
    # print("\n" + "=" * 70, "\n②③ 语义检索(归一化向量的点积=余弦，取 top-k)\n" + "=" * 70)
    def topk(qv, k):
        sims = matrix @ qv[0]                               # 归一化后点积=余弦
        order = sims.argsort()[::-1][:k]
        return [(float(sims[i]), CORPUS[int(i)]) for i in order]
    for q in queries:
        best = topk(embed([q]), k)[0]
        print(f"  查询: {q!r}\n   → [{best[0]:.2f}] {best[1]}")
    return topk


if __name__ == "__main__":
    print(f">>> 设备={DEV}\n")
    matrix = build_index()
    topk = search(matrix, ["how to search text by meaning", "handle data bigger than RAM",
                           "who runs the training loop"])
    # 自检：按意思检索，换说法也命中
    hit = topk(embed(["search by meaning"]), 1)[0][1]
    assert "meaning" in hit, hit
    print("\n✅ Ch5 跑通：批量嵌入 → 归一化向量矩阵 → 余弦 top-k 检索。换说法也能命中。")
    # print("面试：Q 为什么用向量+余弦不用关键词? Q 句向量为什么 mean 要用 mask? Q FAISS 解决什么问题?"
          # " (见 ../面试高频题库.py 四.RAG)")
