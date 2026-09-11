"""
================================================================================
 生产02 · 生产级 RAG：向量数据库(Qdrant/Milvus) + 嵌入服务 + 检索 + 生成
================================================================================
 ⚠ 需要 Qdrant/Milvus 服务 + (可选)GPU 生成，本机跑不了；这是“真上线怎么搭”的架构参考。
   本文件的代码是真实可用的 Python(不是字符串)：import 都写成惰性(用到才导入)，所以
   `python3 生产02_RAG_Qdrant向量库.py` 不会因缺 qdrant-client/sentence-transformers 报错，
   只打印架构说明；把 build_index()/make_app() 放到装好依赖+起了 Qdrant 的环境里就能真跑。
   本地能跑的小版本(内存向量+余弦)见 ../案例2_RAG语义检索问答.py。

 为什么生产 RAG 要用向量数据库(而不是内存里存向量)：
   · 规模：几百万~上亿文档的向量,内存放不下;向量库落盘 + 内存映射,还能横向扩展。
   · 检索快：向量库用 HNSW/IVF 等近似最近邻索引(ANN),百万级毫秒返回,内存暴力算扛不住。
   · 工程化：增量写入/更新/删除、元数据过滤(按时间/权限/来源筛)、持久化、备份、高可用。
   · 美国常用 Pinecone/Weaviate/pgvector/Milvus；中国常用 Milvus(Zilliz,中国团队)。

 生产 RAG 数据流：
   离线：文档 → 切块(chunk) → 嵌入模型 → 向量 + 元数据 → 写入向量库(建索引)
   在线：用户问题 → 嵌入 → 向量库 ANN 检索 top-k → 拼进提示 → LLM 生成带引用的回答

 —— 起 Qdrant(docker) ——
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   pip install qdrant-client sentence-transformers
================================================================================
"""
import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION", "kb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")   # 生产常用强嵌入(bge-m3 中英都好)
DIM = 1024                                              # bge-m3 向量维度
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8001/v1/chat/completions")


# ------------------------------------------------------------------------------
# 嵌入服务：加载一次全程复用(生产里嵌入模型常单独部署成一个批量推理服务)
# ------------------------------------------------------------------------------
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer   # 惰性导入
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def embed(texts):
    # normalize_embeddings=True → 向量归一化，之后用余弦距离检索
    return get_embedder().encode(texts, normalize_embeddings=True)


# ------------------------------------------------------------------------------
# 一、离线：建库 + 灌数据(把文档向量化写进 Qdrant)
# ------------------------------------------------------------------------------
def build_index(docs=None):
    """docs: [{"id","text","source","lang"}, ...]。真实项目从数据湖/wiki/PDF 切块而来。"""
    from qdrant_client import QdrantClient                       # 惰性导入
    from qdrant_client.models import Distance, VectorParams, PointStruct

    docs = docs or [
        {"id": 1, "text": "重置密码：设置>安全>重置密码。", "source": "faq", "lang": "zh"},
        {"id": 2, "text": "上传照片闪退：升级到 v3.2。", "source": "faq", "lang": "zh"},
    ]
    client = QdrantClient(url=QDRANT_URL)

    # 建集合(索引)：指定向量维度 + 距离度量(余弦)。新版 API 用 collection_exists+create_collection
    # (旧的 recreate_collection 已弃用；生产也不该无脑 recreate——那会清空已有数据)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )

    vectors = embed([d["text"] for d in docs])
    # 批量写入：向量 + 元数据(payload，可用于检索时过滤)
    client.upsert(collection_name=COLLECTION, points=[
        PointStruct(id=d["id"], vector=v.tolist(),
                    payload={"text": d["text"], "source": d["source"], "lang": d["lang"]})
        for d, v in zip(docs, vectors)
    ])
    return client


# ------------------------------------------------------------------------------
# 二、在线：检索 + 生成(FastAPI 服务)
# ------------------------------------------------------------------------------
def make_app():
    """返回一个 FastAPI app。用法：uvicorn 生产02_RAG_Qdrant向量库:make_app --factory ..."""
    import httpx
    from fastapi import FastAPI
    from pydantic import BaseModel
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    app = FastAPI(title="生产级 RAG 服务")
    client = QdrantClient(url=QDRANT_URL)

    class Query(BaseModel):
        question: str
        lang: str = "zh"
        top_k: int = 3

    @app.post("/rag")
    async def rag(q: Query):
        # ① 嵌入问题
        qv = embed(q.question).tolist()
        # ② 向量库 ANN 检索 top-k(还能按元数据过滤，如只查中文/某来源/有权限的)
        #    新版 API 用 query_points(旧的 search 已弃用)
        hits = client.query_points(
            collection_name=COLLECTION, query=qv, limit=q.top_k, with_payload=True,
            query_filter=Filter(must=[FieldCondition(key="lang", match=MatchValue(value=q.lang))]),
        ).points
        context = "\n".join(f"[{i+1}] {h.payload['text']}" for i, h in enumerate(hits))
        # ③ 拼进提示，喂 LLM 生成带引用的回答(生成走 vLLM，见 生产01)
        prompt = f"根据下面资料回答问题，并标注引用编号。\n资料:\n{context}\n\n问题: {q.question}"
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(VLLM_URL, json={"model": "qwen",
                "messages": [{"role": "user", "content": prompt}]})
        answer = r.json()["choices"][0]["message"]["content"]
        # 返回答案 + 命中的来源(可溯源，生产 RAG 必须能引用出处)
        return {"answer": answer,
                "sources": [{"score": h.score, "text": h.payload["text"]} for h in hits]}

    return app


# ------------------------------------------------------------------------------
# 三、生产要点
# ------------------------------------------------------------------------------
NOTES = """
 · 切块策略：太大→检索不准&塞不下；太小→丢上下文。常见 200-500 token/块 + 10-20% 重叠。
 · 嵌入模型：中英选 bge-m3 / bge-large；要多语言选 multilingual。嵌入服务可单独部署、批量推理。
 · 混合检索：向量检索(语义) + 关键词检索(BM25) 融合(RRF)，比单一召回更全(生产常用)。
 · 重排(rerank)：召回 top-50 再用 cross-encoder 重排取 top-5，精度更高。
 · 元数据过滤：按权限/时间/租户过滤，避免跨权限泄露(企业 RAG 的红线)。
 · 更新：文档变了要能增量 upsert / 删除对应向量,别全库重建。
 · 监控：召回率、答案是否有据(防幻觉)、端到端延迟。
 · 中国云：Milvus(Zilliz Cloud) / 阿里云向量检索 / 腾讯云向量数据库，接口思路一致。
"""

if __name__ == "__main__":
    print(__doc__)
    print("=== 三、生产要点 ===", NOTES)
    print(">>> 架构参考。真跑步骤：")
    print("    1) docker run -p 6333:6333 qdrant/qdrant   # 起向量库")
    print("    2) pip install qdrant-client sentence-transformers")
    print("    3) python3 -c 'import 生产02_RAG_Qdrant向量库 as m; m.build_index()'  # 离线灌数据")
    print("    4) GPU 机起 vLLM(见 生产01)，再 uvicorn 生产02_RAG_Qdrant向量库:make_app --factory")
    print("    本地小版本见 ../案例2_RAG语义检索问答.py")
