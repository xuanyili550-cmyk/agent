"""
================================================================================
 Chapter 5 挖空练习 · 数据集处理 + 语义搜索（FAISS）
================================================================================
 玩法：
   1) 把每个 ______ 换成正确代码（凭记忆，别翻笔记）
   2) 运行：python3 Chapter5_挖空练习.py
   3) 没填的报 NameError（告诉你漏哪行）；卡住 → 文件底部「答案区」
 目标：把一个小知识库建成 FAISS 索引，用语义搜索找出最相关的文档。
 ⚠ 本机说明：此练习用 datasets 库(Dataset.map/add_faiss_index)，而本机 Python3.14 下 datasets 的
   pickle 坏了(会报 Pickler._batch_setitems)，需在【正常环境(Colab/云)】跑。本机可跑的等价版(纯 numpy
   余弦检索)见 ../实战练习/chapter实战/分章项目/Ch5_进阶_数据工程.py。答案区仍是标准 datasets+FAISS 写法。
================================================================================
"""
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel

corpus = [
    "You can load a dataset offline without internet.",
    "FAISS builds an index to find similar embeddings quickly.",
    "Semantic search finds documents by meaning, not keywords.",
    "Padding makes all sequences in a batch the same length.",
]

ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModel.from_pretrained(ckpt)
model.eval()


def embed(texts):
    # 练习1：编码——补齐、截断、返回 PyTorch 张量
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    # 练习2：CLS 池化——取每句第一个 token([CLS]) 的最后隐藏状态（下标填几？）
    return out.last_hidden_state[:, 0]


# 练习3：从字典建 Dataset（方法名？）
ds = Dataset.from_dict({"text": corpus})
# 练习4：map 出 embeddings 列——每条取 [?] 变成一维向量
ds = ds.map(lambda x: {"embeddings": embed([x["text"]]).cpu().numpy()[0]})
# 练习5：给 embeddings 列建 FAISS 索引（方法名？）
ds.add_faiss_index(column="embeddings")

query = "how to search text by meaning"
q_emb = embed([query]).cpu().numpy()
# 练习6：检索最近的 2 条（方法名？参数：列名, 查询向量, k=?）
scores, samples = ds.get_nearest_examples("embeddings", q_emb, k=2)

print(f"问题：{query}")
for s, t in zip(scores, samples["text"]):
    print(f"  [{s:.1f}] {t}")
print("自检：top1 应命中 'Semantic search finds documents by meaning'")


# ==============================================================================
#  答案区（卡住再看！）
# ------------------------------------------------------------------------------
#  1: padding=True, truncation=True, return_tensors="pt"
#  2: out.last_hidden_state[:, 0]
#  3: Dataset.from_dict({"text": corpus})
#  4: embed([x["text"]]).cpu().numpy()[0]
#  5: ds.add_faiss_index(column="embeddings")
#  6: ds.get_nearest_examples("embeddings", q_emb, k=2)
# ==============================================================================
