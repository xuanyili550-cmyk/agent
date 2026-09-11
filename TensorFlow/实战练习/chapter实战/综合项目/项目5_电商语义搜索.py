"""
================================================================================
 综合项目5 · 电商语义搜索 + 重排（中文版，整合 Ch1 + Ch5 + Ch6）
================================================================================
 用户搜一句话 → 语义检索商品 → 重排 → 返回最相关商品。解决“换个说法就搜不到”的关键词搜索痛点。
 全部用【中文模型/中文商品】：
   [Ch6 分词+嵌入] 商品标题/描述用 bge-small-zh 嵌成向量(中文)。
   [Ch5 语义检索]  用户 query 嵌向量、按余弦召回 top-N(“夏天防晒穿的”能召回“UPF50 防晒短袖”)。
   [Ch1 排序/重排]  召回后再按“query-商品”相关性重排(生产用 cross-encoder rerank)取 top-k。
   为什么两段(召回+重排)：召回要快(向量 ANN 扫全库)、重排要准(cross-encoder 只算 top-N)，快与准兼得。

 bge 中文嵌入：CLS 池化 + 查询加前缀(和英文 mean 池化不同)。
 本地跑：python3 项目5_电商语义搜索.py
================================================================================
"""
import torch
import torch.nn.functional as F

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："      # bge 中文检索：查询加前缀


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


# 商品库(标题；真实项目还有类目/价格/销量等做多路召回+排序)
PRODUCTS = [
    "UPF50+ 轻薄防晒长袖衬衫 夏季防紫外线",
    "防水登山徒步鞋 带脚踝支撑",
    "主动降噪无线头戴式耳机",
    "316 不锈钢保温杯 1L 24小时保冷保热",
    "速干透气跑步短裤 带拉链口袋",
    "宽檐草编遮阳帽 沙滩户外防晒",
    "机械键盘 RGB 背光 热插拔轴体",
]


class Search:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        self.dev = pick_device()
        e = "BAAI/bge-small-zh-v1.5"
        self.tok = AutoTokenizer.from_pretrained(e)
        self.emb = AutoModel.from_pretrained(e).to(self.dev).eval()
        self.vecs = self.embed(PRODUCTS)          # 离线：商品向量索引(生产放向量库)

    def embed(self, texts, is_query=False):
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        enc = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            v = self.emb(**enc).last_hidden_state[:, 0]   # [Ch5] bge 用 CLS 池化
        return F.normalize(v, p=2, dim=1)

    def recall(self, query, top_n=4):
        sims = (self.embed([query], is_query=True) @ self.vecs.T)[0]   # [Ch5] 向量召回
        idx = sims.argsort(descending=True)[:top_n]
        return [(float(sims[i]), PRODUCTS[i]) for i in idx]

    def rerank(self, query, candidates):
        # 简化重排：这里仍用向量分(演示两段结构)。生产用 cross-encoder(bge-reranker)把
        # query 和每个候选“一起”过一遍打分,比双塔向量更准。
        return sorted(candidates, key=lambda x: -x[0])

    def search(self, query, top_k=3):
        cands = self.recall(query, top_n=5)          # ① 召回(快)
        ranked = self.rerank(query, cands)           # ② 重排(准)
        return ranked[:top_k]


def main():
    s = Search()
    print(f">>> 设备={s.dev}  商品库 {len(PRODUCTS)} 个（全中文模型）\n")
    for q in ["夏天大太阳天穿的衣服",       # 期望召回 防晒衣/草帽
              "徒步时能保持水冰凉的东西",     # 保温杯
              "打游戏用的键盘"]:            # 机械键盘
        print(f"搜索: {q}")
        for score, p in s.search(q):
            print(f"  [{score:.2f}] {p}")
        # print()
    # 自检：热天穿的应召回“防晒”相关
    assert any("防晒" in p for _, p in s.search("夏天大太阳天穿的衣服"))
    print("✅ 中文电商语义搜索跑通：向量召回(Ch5)+重排(Ch1)，换说法也能搜到对的商品。")


# ==============================================================================
# 生产要点 + 面试题
# ==============================================================================
# 生产：· 多路召回：语义(向量)+关键词(BM25，中文要先分词)+类目/品牌过滤，融合(RRF)。
#      · 重排：cross-encoder(bge-reranker)对 top-N 精排；再叠业务排序(销量/价格/个性化)。
#      · 向量库：Milvus/Qdrant，商品增删改要实时同步向量。
#      · 冷启动/长尾：向量召回对长尾 query 友好(关键词搜不到的能召回)。
#      · 评估：召回率@k、NDCG、点击率/转化率(线上 A/B)。
# 面试：Q 为什么召回+重排两段？A 召回要快扫全库(双塔向量),重排要准(cross-encoder 算 top-N),兼顾快与准。
#      Q 双塔(bi-encoder) vs 交互(cross-encoder) 区别？A 双塔分别编码可预存向量、快，适合召回；
#        交互把 query+doc 一起编码、准但慢,适合重排。
#      Q 中文语义搜索为什么要用中文嵌入？A 英文嵌入语义空间对不上中文；bge-zh/text2vec 才准。

if __name__ == "__main__":
    main()
