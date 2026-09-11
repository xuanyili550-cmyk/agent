"""
================================================================================
 Agent Course · Chapter 3 · Agentic RAG（学习笔记，可运行真代码）
================================================================================
 普通 RAG：固定“检索一次 → 塞提示 → 回答”。Agentic RAG：把【检索】做成工具交给 Agent，
 让 LLM 自己判断“要不要查、查什么、够不够、要不要再查”。更灵活、更准。
 整合：检索用 Ch5 的中文语义检索(bge-small-zh，CLS 池化+查询前缀)；生成用 mlx-lm(本地真跑)。
 内容：① Agentic RAG vs 普通 RAG  ② 检索工具  ③ Agent 决定检索+生成(真跑)  ④ 多轮记忆(概念)
 跑：python3 chapter3_AgenticRAG_学习笔记.py
================================================================================
"""
import re

_M = {}
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
KB = [
    "免费版包含 5GB 存储和基础支持。",
    "专业版每月 12 美元，含 1TB 存储、优先邮件支持和团队协作。",
    "企业版提供无限存储、专属客户经理和 SSO 单点登录。",
    "数据存储用 AES-256 加密，传输用 TLS 1.3，已通过 SOC 2 Type II 合规。",
    "重置密码：设置 > 安全 > 重置密码，按邮件链接操作。",
]


def _dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ② 检索工具(bge 中文语义检索)
def _embed(texts, is_query=False):
    import torch
    import torch.nn.functional as F
    if "emb" not in _M:
        from transformers import AutoTokenizer, AutoModel
        n = "BAAI/bge-small-zh-v1.5"
        _M["et"] = AutoTokenizer.from_pretrained(n)
        _M["em"] = AutoModel.from_pretrained(n).to(_dev()).eval()
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    enc = _M["et"](texts, padding=True, truncation=True, return_tensors="pt").to(_dev())
    with torch.no_grad():
        v = _M["em"](**enc).last_hidden_state[:, 0]      # CLS 池化
    return F.normalize(v, p=2, dim=1)


def search_kb(query, top_k=2):
    """检索工具：在知识库里语义检索 top-k(Agent 会调它)。"""
    if "kv" not in _M:
        _M["kv"] = _embed(KB)
    sims = (_embed([query], is_query=True) @ _M["kv"].T)[0]
    idx = sims.argsort(descending=True)[:top_k]
    return [(float(sims[i]), KB[i]) for i in idx]


# ③ Agent 大脑(mlx-lm)决定检索 + 生成
def llm(p, n=100):
    if "m" not in _M:
        from mlx_lm import load
        _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import generate
    t = _M["t"].apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
    return generate(_M["m"], _M["t"], prompt=t, max_tokens=n, verbose=False).strip()


def agentic_rag(question, threshold=0.5):
    """Agentic：检索 → 判断够不够 → (换词再检索) → 基于资料生成带引用回答。"""
    hits = search_kb(question, top_k=2)
    print(f"  🔍 检索1：{[round(s, 2) for s, _ in hits]}")
    if hits[0][0] < threshold:                            # Agent 判断“不够好”→ 换个说法再查(模拟)
        alt = question.replace("多少钱", "价格").replace("安全吗", "加密 合规")
        if alt != question:
            hits2 = search_kb(alt, top_k=2)
            print(f"  🔁 换词再检索：{alt} → {[round(s, 2) for s, _ in hits2]}")
            if hits2[0][0] > hits[0][0]:
                hits = hits2
    context = "\n".join(f"[{i+1}] {c}" for i, (_, c) in enumerate(hits))
    ans = llm(f"只根据资料回答，没有就说不知道。\n资料：\n{context}\n问题：{question}\n回答：")
    return ans, hits


if __name__ == "__main__":
    print("① Agentic RAG：把检索做成工具，LLM 自己决定查什么/查几次(vs 普通 RAG 固定查一次)")
    print("② 检索工具自检(bge 中文)：", search_kb("专业版多少钱")[0][1][:30], "...")
    assert any("12" in c for _, c in search_kb("专业版价格"))
    assert any("加密" in c for _, c in search_kb("你们数据安全吗"))   # 换说法也命中
    print("③ Agent 决定检索 + 生成(mlx-lm 真跑)：")
    for q in ["专业版套餐多少钱？", "你们的数据安全合规吗？"]:
        print(f"\n❓ {q}")
        ans, hits = agentic_rag(q)
        print(f"  💬 回答：{ans[:100]}")
    print("\n④ 多轮记忆(概念)：smolagents reset=False / llama-index Context / langgraph messages。")
    print("\n✅ Chapter3 跑通：检索工具(Ch5 bge) + Agentic 检索决策 + mlx-lm 生成带引用回答。")
    print("面试：Q Agentic RAG vs 普通 RAG? Q 检索不准怎么排查? Q 怎么防幻觉?(只据资料+标引用)")
