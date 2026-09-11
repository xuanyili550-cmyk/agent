"""
================================================================================
 Chapter 5 · Datasets 库 + 语义搜索(FAISS) —— 系统学习笔记
================================================================================
 配套 HuggingFace 课程「The Datasets library」章节（对应你的 4 个文件）。
 主线：怎么把各种数据喂进模型 + 怎么用「语义搜索」找相似文本(=RAG 的检索核心)。

 ★ 本章最有价值的是「语义搜索」——它就是 RAG(检索增强生成)的前半段，
   也是你作品集项目2的地基。所以笔记以它为重点。

 本文件结构（对应你的 4 个文件）：
   第 1 部分  加载数据集：本地/远程/各种格式        [dataset.py]
   第 2 部分  切片切块处理：filter/map/rename/sort   [CutIntoPieces.py]
   第 3 部分  大数据：内存映射 + 流式 streaming      [zstandard.py]
   第 4 部分  ★语义搜索：嵌入 + FAISS 索引 + 检索    [FAISS.py] ← 重点
   第 5 部分  ★★语义搜索 vs 关键词搜索：为什么它强
   第 6 部分  常见报错速查（cuda、faiss 未装等）

 术语速记：
   embedding(嵌入)   把一段文本变成一串数字(向量)，语义相近的文本向量也相近
   pooling(池化)     把模型输出的多个 token 向量，压成「一句话一个向量」
   FAISS             Facebook 的向量检索库，能在百万级向量里秒找最相似的
   语义搜索          按「意思」找，而不是按「字面关键词」找
   RAG               检索(找相关文档) + 生成(LLM 基于文档回答)
================================================================================
"""

# ==============================================================================
# 第 1 部分：加载数据集（本地 / 远程 / 各种格式）
# ==============================================================================
# load_dataset 能吃各种格式，靠第一个参数指定类型：
#   CSV/TSV   load_dataset("csv",  data_files="my.csv")          # TSV 加 delimiter="\t"
#   文本      load_dataset("text", data_files="my.txt")
#   JSON      load_dataset("json", data_files="my.jsonl")        # 嵌套的加 field="data"
#   pandas    load_dataset("pandas", data_files="my.pkl")
#
# from datasets import load_dataset
# # 本地单文件
# ds = load_dataset("json", data_files="SQuAD_it-train.json", field="data")
# # 多文件分 train/test
# data_files = {"train": "train.json", "test": "test.json"}
# ds = load_dataset("json", data_files=data_files, field="data")
# # 远程 URL（甚至压缩包 .gz/.zst 都能直接读，不用先解压）
# url = "https://github.com/.../SQuAD_it-train.json.gz"
# ds = load_dataset("json", data_files=url, field="data")
# # Hub 上的现成数据集
# ds = load_dataset("nyu-mll/glue", "mrpc")   # 你在 Ch2 用过


# ==============================================================================
# 第 2 部分：切片切块处理（数据清洗的常用招式）
# ==============================================================================
# 拿到原始数据后，通常要清洗。核心方法：
#
#   .filter(fn)                 按条件筛行；fn 返回 True 的留下
#     drug_dataset.filter(lambda x: x["condition"] is not None)
#
#   .map(fn)                    对每行做变换/加新列；batched=True 更快
#     ds.map(lambda x: {"len": len(x["review"].split())})
#
#   .rename_column(old, new)    改列名
#     ds.rename_column("Unnamed: 0", "patient_id")
#
#   .remove_columns([...])      删列
#
#   .shuffle(seed=42).select(range(1000))   打乱 + 取子集（快速试验用）
#
#   .unique("列")               某列去重后的值（可用来校验，如 ID 是否唯一）
#
#   .sort("列") / .train_test_split(test_size=0.1)   排序 / 切训练验证集
#
#   set_format("pandas")        临时切成 pandas，方便用 df 的操作（如 explode）
#     df = ds[:]                # 取成 DataFrame
#     df.explode("comments")    # 把一行里的列表拆成多行
#   Dataset.from_pandas(df)     # 处理完再转回 Dataset


# ==============================================================================
# 第 3 部分：大数据 —— 内存映射 + 流式 streaming
# ==============================================================================
# 问题：数据集比内存还大怎么办？（如几十 GB 的 PubMed/Pile）
#
# 招式1：内存映射(memory mapping) —— datasets 底层用 Apache Arrow，
#        数据留在磁盘、按需读，不全load进内存。所以你能处理比 RAM 大的数据集。
#        （zstandard.py 里用 psutil 量了内存占用、timeit 测了遍历速度）
#
# 招式2：流式 streaming=True —— 边下边用，完全不落盘、不占内存：
#   stream = load_dataset("json", data_files=url, split="train", streaming=True)
#   next(iter(stream))                 # 逐条取
#   stream.map(lambda x: tokenizer(x["text"]))     # 流式也能 map
#   stream.shuffle(buffer_size=10_000, seed=42)    # 缓冲区内打乱(不是全局)
#   stream.take(5)      # 取前5条        stream.skip(1000)  # 跳过前1000条
#   interleave_datasets([s1, s2])       # 交错合并多个流
# 用途：训练超大模型时，数据根本装不下，只能流式喂。


# ==============================================================================
# 第 4 部分：★语义搜索 —— 嵌入 + FAISS 索引 + 检索（RAG 检索核心）
# ==============================================================================
# 目标：给一个问题，从一堆文档里找出「意思最接近」的几条。三步：
#   ① 把每条文档变成向量(embedding)
#   ② 建 FAISS 索引（一种能快速找相似向量的数据结构）
#   ③ 把问题也变成向量，去索引里找最近的 k 条

from transformers import AutoTokenizer, AutoModel
import torch

# 选一个「专为语义搜索训练」的嵌入模型（普通 bert 效果差）
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

# Mac 用 mps（原课程写死 cuda 会崩）
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# ---- pooling：把「一句话的多个 token 向量」压成「一个句向量」----
# 这个模型用 CLS 池化：取第一个 token([CLS]) 的最后隐藏状态代表整句
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]
#   （另一种常见是 mean pooling：按 attention_mask 加权平均所有 token，
#     如 all-MiniLM-L6-v2 用的就是它。用哪种取决于模型是怎么训练的。）

def get_embeddings(text_list):
    enc = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    return cls_pooling(model(**enc))     # 返回 [句子数, 768] 的向量

# ---- 建 FAISS 索引并检索（datasets 内置了 faiss 支持，超方便）----
# embeddings_dataset = comments_dataset.map(
#     lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]})
# embeddings_dataset.add_faiss_index(column="embeddings")   # 一行建索引
#
# question = "How can I load a dataset offline?"
# q_emb = get_embeddings([question]).cpu().detach().numpy()
# scores, samples = embeddings_dataset.get_nearest_examples("embeddings", q_emb, k=5)
# # scores：相似度分数（越大越像）；samples：最匹配的 5 条文档
#
# 注意：跑真实数据集要 pip install faiss-cpu（见第6部分），且 device 别写死 cuda。


# ==============================================================================
# 第 5 部分：★★语义搜索 vs 关键词搜索 —— 为什么它更强
# ==============================================================================
# 关键词搜索(如 Ctrl+F / 传统数据库 LIKE)：只匹配「字面相同」的词。
#   查 "how to load data offline" → 必须文档里也有这些词，换个说法就找不到。
#
# 语义搜索：匹配「意思相近」。
#   查 "怎么离线加载数据集" → 能找到写着 "load dataset without internet" 的文档，
#   哪怕一个词都不一样，因为它们的向量很接近。
#
# 这就是为什么 RAG 用语义搜索：用户问法千变万化，靠语义才能稳稳找到相关资料。
# （你在 Ch1 综合案例场景2 已经手动做过一次：句向量 + 余弦相似度，就是这个原理。
#   Ch5 把它升级成「用 FAISS 索引，百万级文档也能秒查」的工业级方案。）


# ==============================================================================
# 第 6 部分：常见报错速查
# ==============================================================================
# ┌────────────────────────────────────────────┬──────────────────────────────┐
# │ 报错                                          │ 原因 & 解决                   │
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ RuntimeError: ... CUDA / torch.device("cuda")│ Mac 没 N卡。→ 改             │
# │  报 device 相关错                              │ torch.device("mps" if ...    │
# │                                              │ else "cpu")                  │
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ ModuleNotFoundError: No module named 'faiss' │ 没装 faiss。                  │
# │                                              │ → pip install faiss-cpu       │
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ add_faiss_index 报 embeddings 维度不齐         │ 每条 embedding 要同长度        │
# │                                              │ (取 [0] 变成一维向量)          │
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ 语义搜索结果很差 / 不相关                       │ 用了普通 bert 而非句向量模型。  │
# │                                              │ → 用 sentence-transformers/*  │
# │                                              │ 系列，且 pooling 方式要匹配模型 │
# └────────────────────────────────────────────┴──────────────────────────────┘
#
# 一句话总结本章：
#   Datasets 库负责「把各种数据高效喂进来」；语义搜索(嵌入+FAISS)负责
#   「按意思找相似文本」——后者是 RAG 的检索核心，也是你最该练熟的一块。
