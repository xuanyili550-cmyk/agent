"""
================================================================================
 Agent 进阶案例3 · 混合检索 RAG + LangGraph（对标 ai-agents-from-zero 核心，本机真跑）
================================================================================
 补齐"企业级检索"短板：不是单路向量，而是【向量(语义) + BM25(关键词)】双路召回 → RRF 融合 →
 重排 → LangGraph 编排 → mlx 生成带引用回答。企业 RAG 的标准做法。
 技术：bge-small-zh 向量检索 + rank_bm25 关键词检索 + RRF 倒数排名融合 + 简单重排 +
      LangGraph 状态机(检索→融合→生成) + mlx-lm 生成。cross-encoder 重排给生产参考。
 跑：python3 案例3_混合检索RAG_LangGraph.py
================================================================================
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

_M = {}
QP = "为这个句子生成表示以用于检索相关文章："
KB = [
    "重置密码：进入 设置 > 安全 > 重置密码，按邮件链接操作。",
    "专业版每月 12 美元，包含 1TB 存储和优先邮件支持。",
    "企业版提供无限存储、专属客户经理和 SSO 单点登录集成。",
    "数据在存储时用 AES-256 加密，传输用 TLS 1.3，已通过 SOC 2 Type II 合规。",
    "上传照片闪退：请升级 App 到 v3.2 或更高版本，该问题已修复。",
    "免费版包含 5GB 存储和基础邮件支持。",
]


def _dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---- 双路召回 ----
def _vector_topk(query, k=4):
    import torch
    import torch.nn.functional as F
    if "emb" not in _M:
        from transformers import AutoTokenizer, AutoModel
        _M["et"] = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
        _M["em"] = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5").to(_dev()).eval()

    def emb(ts, q=False):
        if q:
            ts = [QP + t for t in ts]
        enc = _M["et"](ts, padding=True, truncation=True, return_tensors="pt").to(_dev())
        with torch.no_grad():
            v = _M["em"](**enc).last_hidden_state[:, 0]
        return F.normalize(v, p=2, dim=1)
    if "kv" not in _M:
        _M["kv"] = emb(KB)
    sims = (emb([query], q=True) @ _M["kv"].T)[0]
    return [int(i) for i in sims.argsort(descending=True)[:k]]


def _bm25_topk(query, k=4):
    from rank_bm25 import BM25Okapi
    import re
    def tok(t):                                    # 中文按字 + 英数按词(简单分词够 BM25 用)
        return re.findall(r"[a-zA-Z0-9]+|[一-鿿]", t)
    if "bm25" not in _M:
        _M["bm25"] = BM25Okapi([tok(d) for d in KB])
    scores = _M["bm25"].get_scores(tok(query))
    return [int(i) for i in sorted(range(len(KB)), key=lambda i: -scores[i])[:k]]


def _rrf(rank_lists, kconst=60):
    """RRF 倒数排名融合：每个文档得分 = Σ 1/(k + 排名)。"""
    score = {}
    for lst in rank_lists:
        for rank, idx in enumerate(lst):
            score[idx] = score.get(idx, 0) + 1 / (kconst + rank)
    return sorted(score, key=lambda i: -score[i])


# ---- LangGraph 状态机 ----
class State(TypedDict):
    query: str
    vec: list
    bm25: list
    fused: list
    context: list
    answer: str
    trace: list


def n_vector(s):
    return {"vec": _vector_topk(s["query"]), "trace": s["trace"] + ["向量召回(bge)"]}


def n_bm25(s):
    return {"bm25": _bm25_topk(s["query"]), "trace": s["trace"] + ["关键词召回(BM25)"]}


def n_fuse(s):
    fused = _rrf([s["vec"], s["bm25"]])[:3]           # RRF 融合取 top-3
    # 简单重排：把与 query 有字面重合的往前提(近似 cross-encoder 的作用)
    fused.sort(key=lambda i: -sum(1 for c in set(s["query"]) if c in KB[i]))
    return {"fused": fused, "context": [KB[i] for i in fused],
            "trace": s["trace"] + [f"RRF 融合+重排 → top{len(fused)}"]}


def n_generate(s):
    if "m" not in _M:
        from mlx_lm import load
        _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import generate
    ctx = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(s["context"]))
    ans = generate(_M["m"], _M["t"], prompt=_M["t"].apply_chat_template(
        [{"role": "user", "content": f"只根据资料回答并标引用编号。\n资料：\n{ctx}\n问题：{s['query']}\n回答："}],
        add_generation_prompt=True), max_tokens=80, verbose=False).strip()
    return {"answer": ans, "trace": s["trace"] + ["mlx 生成带引用回答"]}


def build_graph():
    g = StateGraph(State)
    for name, fn in [("vector", n_vector), ("bm25", n_bm25), ("fuse", n_fuse), ("generate", n_generate)]:
        g.add_node(name, fn)
    g.add_edge(START, "vector")
    g.add_edge("vector", "bm25")           # 顺序双路(也可并行 fan-out)
    g.add_edge("bm25", "fuse")
    g.add_edge("fuse", "generate")
    g.add_edge("generate", END)
    return g.compile()


def ask(query):
    return build_graph().invoke({"query": query, "vec": [], "bm25": [], "fused": [],
                                 "context": [], "answer": "", "trace": []})


# —— cross-encoder 重排(生产)参考：召回后精排，比 RRF 更准(需下 reranker 模型) ——
CROSS_ENCODER_REF = '''
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")
pairs = [(query, KB[i]) for i in candidates]
scores = reranker.predict(pairs)          # query-doc 一起过一遍打分
reranked = [c for _, c in sorted(zip(scores, candidates), reverse=True)][:5]
'''

if __name__ == "__main__":
    for q in ["专业版多少钱", "数据加密合规吗", "照片上传闪退怎么办"]:
        r = ask(q)
        # print("─" * 64)                                              # 去装饰分隔线
        print(f"❓ {q}")
        # 自检噪音：中间召回 idx + 逐条 trace(向量→BM25→RRF融合→生成)，省略以减少输出
        # print(f"   向量召回 idx={r['vec']}  BM25召回 idx={r['bm25']}")
        # for step in r["trace"]:
        #     print("   ·", step)
        print(f"   📎 融合命中: {r['context'][0][:30]}...")
        print(f"   💬 {r['answer'][:60]}")
    r = ask("专业版多少钱")
    assert any("12" in c for c in r["context"])           # 混合检索命中价格条
    assert r["vec"] and r["bm25"] and r["fused"]
    print("\n✅ 案例3 跑通：混合检索(向量+BM25+RRF+重排)+LangGraph 编排+mlx 生成，对标 ai-agents-from-zero。")
