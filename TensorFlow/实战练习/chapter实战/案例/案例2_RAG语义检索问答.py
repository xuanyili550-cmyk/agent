"""
================================================================================
 案例2 · RAG 语义检索问答（整合 Ch5 + Ch6）—— 完整可跑，照着手写练熟
================================================================================
 目标：搭一个“按意思检索”的知识库问答(RAG 的检索前半段)。
 整合了哪几章、每步为什么用这个技术：
   [Ch6 分词器]  把文本切成 token 喂给嵌入模型；padding 对齐、attention_mask 标出真实 token。
   [Ch5 嵌入]    用句向量模型把每段文本变成一个向量：语义相近 → 向量也相近。
   [Ch5 mean 池化] 为什么用 attention_mask 加权平均——[PAD] 位置是废料，直接平均会污染句向量。
   [Ch5 余弦检索] 为什么归一化后点积=余弦相似度——只看“方向(语义)”不看“长度(篇幅)”，一次矩阵乘算完。
   [为什么要 RAG] LLM 不知道你的私有知识；先“按意思”检索到相关文档，再喂给 LLM 回答，
                 既准确又能引用来源，还免重新训练模型。

 直接运行：python3 案例2_RAG语义检索问答.py     # 几秒，下 MiniLM(~90MB) 一次
================================================================================
"""
import math
import re
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ============================================================================
# [知识点·混合检索] 向量检索(语义)有个短板：对“专有名词/编号/精确关键词”不敏感
#   (如产品型号 "v3.2"、"2FA"、订单号)——语义相近但字面不同的会命中，字面必须精确匹配的反而漏。
#   所以生产 RAG 常用“混合检索”：向量(管语义) + BM25(管关键词)，再融合两路结果。
# BM25 = 经典词频检索：词在本文档出现越多(TF)、在整个语料越稀有(IDF)得分越高；
#   还对长文档做归一化(b)、对高频词做饱和(k1)，比朴素 TF-IDF 更稳。下面手写一个精简版。
# ============================================================================
def _tok(t):
    return re.findall(r"[a-z0-9]+", t.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [_tok(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        df = Counter(w for d in self.docs for w in set(d))       # 每个词出现在多少篇文档
        # IDF：越稀有(df小)权重越大 → 稀有词命中比常见词(the/a)更有信息量
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def scores(self, query):
        out = []
        for d in self.docs:
            tf, dl, s = Counter(d), len(d), 0.0
            for w in _tok(query):
                if w not in self.idf:
                    continue
                f = tf[w]
                # BM25 主公式：TF 饱和(k1) + 文档长度归一化(b)
                s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


def rrf_fuse(rank_lists, k=60):
    """[知识点·RRF 融合] 倒数排名融合：把多路检索的“排名”(而非分数,量纲不可比)相加。
    某文档在各路排得越靠前，1/(k+rank) 越大 → 综合分越高。工业界融合向量+BM25 的默认做法。"""
    fused = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=fused.get, reverse=True)


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


# 一个小知识库(每条=一段可被检索的文档)。真实项目里这些来自你的文档/工单/wiki。
KNOWLEDGE = [
    "To reset your password, go to Settings > Security > Reset Password and follow the email link.",
    "The mobile app crashes when uploading photos; upgrade to version 3.2 or later to fix it.",
    "Refunds for duplicate charges are processed within 3-5 business days after verification.",
    "To contact a human agent, type 'agent' in the chat or call the hotline from 9am to 6pm.",
    "Free accounts can store up to 5GB; upgrade to Pro for 1TB of storage.",
    "Two-factor authentication can be enabled in Settings > Security > 2FA.",
]


def main():
    device = pick_device()
    ckpt = "sentence-transformers/all-MiniLM-L6-v2"
    # [自检] 设备/嵌入模型/知识库条数 配置回显（噪音，已静音）

    # ---- [Ch6] 分词器 + [Ch5] 嵌入模型(无任务头，只要 token 向量) ----
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt).to(device).eval()

    def embed(texts):
        # [Ch6] 分词：padding 对齐、truncation 防超长、返回张量
        enc = tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc).last_hidden_state          # (B, L, H) 每个 token 一个向量
        # [Ch5] mean 池化(用 mask 加权)：只平均真实 token，避开 [PAD] 废料
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out * mask).sum(1)
        vec = summed / mask.sum(1).clamp(min=1e-9)        # → (B, H) 一段一个句向量
        # [Ch5] L2 归一化：归一化后两向量点积 = 余弦相似度
        return F.normalize(vec, p=2, dim=1)

    # 预先把知识库编码成向量库(真实项目里存进向量数据库/FAISS 索引，这里放内存)
    kb_vecs = embed(KNOWLEDGE)

    # ---- [知识点·验证] 归一化后“点积==余弦相似度”，不是口号，跑一下证明 ----
    # 余弦 cos = (a·b)/(|a||b|)；两向量都 L2 归一化后 |a|=|b|=1，所以 a·b 直接就是 cos。
    # 好处：整个知识库一次矩阵乘就算完所有相似度，比逐对算 cos 快得多。
    a, b = kb_vecs[0], kb_vecs[1]
    dot = float(a @ b)
    cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
    # [自检] 归一化后 点积==余弦相似度（相等，故整库可用一次矩阵乘算完）—— 验证性输出，已静音

    def retrieve(query, top_k=2):
        q = embed([query])                                # 只编码查询
        sims = (q @ kb_vecs.T)[0]                         # [Ch5] 余弦相似度(点积)
        idx = sims.argsort(descending=True)[:top_k]
        return [(float(sims[i]), KNOWLEDGE[i]) for i in idx]

    # 故意用“换了说法”的问题：证明是按意思检索，不是按关键词
    for query in ["I forgot my login credentials",                  # ~ 改密码那条
                  "My photos won't upload and the app closes itself", # ~ 崩溃那条
                  "how much can I store for free"]:                   # ~ 存储那条
        print(f"\n问: {query}")
        for score, doc in retrieve(query):
            print(f"  [{score:.2f}] {doc}")

    # 自检：换了说法也命中对的文档(按语义)
    assert "password" in retrieve("I forgot my login credentials")[0][1].lower()
    assert "crash" in retrieve("My photos won't upload and the app closes itself")[0][1].lower()

    # ---- [知识点·混合检索演示] 向量 vs BM25 vs RRF 融合 ----
    # 用一个“精确关键词”查询，向量检索容易被语义带偏，BM25 靠字面命中；融合两全其美。
    bm25 = BM25(KNOWLEDGE)
    query = "how to enable 2FA"                            # "2FA" 是精确术语，考验字面匹配
    # [自检] 混合检索演示 分节标题（噪音，已静音）  查询: "how to enable 2FA"
    vec_sims = (embed([query]) @ kb_vecs.T)[0]
    vec_rank = vec_sims.argsort(descending=True).tolist()          # 向量路排名(文档下标)
    bm_scores = bm25.scores(query)
    bm_rank = sorted(range(len(KNOWLEDGE)), key=lambda i: bm_scores[i], reverse=True)  # BM25 路排名
    fused = rrf_fuse([vec_rank, bm_rank])                          # RRF 融合
    print(f"    向量Top1: {KNOWLEDGE[vec_rank[0]][:48]}...")
    print(f"    BM25 Top1: {KNOWLEDGE[bm_rank[0]][:48]}...  (靠 '2FA' 字面精确命中)")
    print(f"    RRF 融合Top1: {KNOWLEDGE[fused[0]][:48]}...")

    print("\n✅ RAG 检索跑通：语义检索 + BM25 + RRF 融合，并验证了点积=余弦。")


# ==============================================================================
# 更多知识点(纯概念，检索质量决定 RAG 上限)
# ==============================================================================
# · [切块策略] 长文档要切块(chunk)再嵌入：块太大→检索不精准&塞不进上下文；太小→丢上下文。
#   经验值 200-500 token + 10~20% 重叠(overlap，防信息被切断在边界)；表格/代码要特殊处理。
# · [重排 rerank] 两阶段检索：先用向量/BM25 粗召回 top-50(要快、召回全)，再用 cross-encoder
#   (如 bge-reranker，把 query+doc 拼一起打分，比双塔更准但慢)精排取 top-5。精度大幅提升。
# · [Top-k 选择] k 太小可能漏关键块;太大→噪声多、提示变长、成本高、还可能"迷失在中间"。常 3-5,
#   配重排后可更小。可按相似度阈值动态截断(低于阈值就丢，宁缺毋滥)。
# · [防幻觉] 提示里强制"只根据给定资料回答、无据就说'不知道'、每句标引用编号";答案再做"是否有据"
#   校验;召回为空/相似度过低就走兜底(转人工/说查不到)，别让 LLM 硬编。
# · [下一步·RAG 后半段] 把检索到的块拼进提示喂 LLM 生成带引用的回答，见 综合项目/项目2。

# ==============================================================================
# 面试题(RAG/检索)
# ==============================================================================
# Q1: 语义检索为什么用向量+余弦，不用关键词匹配？各自短板？
# A : 关键词只能字面匹配，换说法就搜不到;向量把语义相近的映射到相近方向，按余弦"按意思找"。
#     但向量对精确术语/编号不敏感 → 生产用"混合检索(向量+BM25)+RRF 融合"互补。
# Q2: 为什么归一化后点积就等于余弦相似度？有什么好处？
# A : cos=(a·b)/(|a||b|)，L2 归一化后 |a|=|b|=1 故点积=余弦。好处:整库一次矩阵乘算完所有相似度，
#     且只看方向(语义)不受向量长度(篇幅)影响。
# Q3: RAG 效果不好怎么系统排查？
# A : 顺链路查:切块质量(大小/重叠)→嵌入模型是否匹配领域&语言→是否该上混合检索/重排→query 改写
#     (口语问题改成检索友好)→top-k 与阈值→提示是否约束"据实回答"。检索是上限，生成只是下限。
# Q4: rerank 为什么能提升精度？和向量检索什么关系？
# A : 向量是"双塔"(query/doc 各自编码,快但粗);cross-encoder 把 query+doc 拼一起过一遍模型,能建模
#     细粒度交互,更准但慢。故两阶段:向量粗召回 top-50 → cross-encoder 精排 top-5,兼顾快与准。
# Q5: 怎么防止 RAG 幻觉？
# A : 提示强制"只据给定资料回答、无据说不知道、标引用";答案做有据校验;召回空/相似度低走兜底;
#     必要时让模型输出引用编号供前端高亮溯源。检索不到就别硬答。


if __name__ == "__main__":
    main()
