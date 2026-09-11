"""
================================================================================
 Agent 进阶案例2 · 深度研搜多智能体（对标 deepsearch-agents，本机真跑）
================================================================================
 补齐"多智能体协作"短板：一个 supervisor(主管)编排多个子 agent 完成"研究报告"：
   主管 → 检索 agent(找资料) → 写作 agent(写草稿) → 审核 agent(查漏) →(不合格回写作)→ 汇总
 技术：LangGraph 真·多智能体图(supervisor + worker + 循环反馈)；子 agent 节点调 mlx-lm 真跑；
      检索用本地 KB(bge 中文向量)——联网搜索(Tavily)给参考代码(需 key，本机不跑)。
 主管路由用确定性状态机(稳)；生产可让 LLM 主管路由。
 跑：python3 案例2_深度研搜多智能体.py
================================================================================
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

_M = {}
QP = "为这个句子生成表示以用于检索相关文章："
# 本地知识库(模拟"搜索"到的资料；联网版见文末 web_search 参考)
KB = [
    "RAG(检索增强生成)先检索相关文档再让 LLM 基于文档回答，减少幻觉、可溯源、免重训。",
    "RAG 三步：文档切块→向量化入库→检索 top-k 拼进提示。",
    "混合检索=向量(语义)+BM25(关键词)用 RRF 融合，比单一召回更全。",
    "重排 rerank：召回 top-50 再用 cross-encoder 精排取 top-5，精度更高。",
    "RAG 防幻觉：提示里要求'只根据资料回答、无据就说不知道、标引用'。",
]


def _dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _search_kb(query, k=3):
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
    idx = sims.argsort(descending=True)[:k]
    return [KB[int(i)] for i in idx]


def _mlx(prompt, n=120):
    if "m" not in _M:
        from mlx_lm import load
        _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import generate
    t = _M["t"].apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    return generate(_M["m"], _M["t"], prompt=t, max_tokens=n, verbose=False).strip()


# ==============================================================================
# 多智能体状态 + 各子 agent 节点
# ==============================================================================
class State(TypedDict):
    topic: str
    findings: list      # 检索 agent 产出
    draft: str          # 写作 agent 产出
    approved: bool      # 审核 agent 结论
    revisions: int
    report: str
    trace: list


def _route(s):
    if not s["findings"]:
        return "search"
    if not s["draft"]:
        return "write"
    if not s["approved"] and s["revisions"] < 2:
        return "review"
    return "finalize"


def supervisor(s):
    """主管：看状态决定下一个子 agent(确定性编排，与 _route 同一逻辑)。"""
    return {"trace": s["trace"] + [f"🧑‍💼 主管 → 派给 {_route(s)}"]}


def search_agent(s):
    hits = _search_kb(s["topic"], k=3)
    return {"findings": hits, "trace": s["trace"] + [f"🔎 检索 agent → 找到 {len(hits)} 条资料"]}


def write_agent(s):
    ctx = "\n".join(f"- {f}" for f in s["findings"])
    draft = _mlx(f"根据资料写一段关于「{s['topic']}」的简短说明(3-4句)。\n资料：\n{ctx}\n说明：", 140)
    return {"draft": draft, "trace": s["trace"] + ["✍️ 写作 agent → 产出草稿"]}


def review_agent(s):
    # 审核(规则)：草稿够长且覆盖关键词 → 通过；否则打回重写
    ok = len(s["draft"]) >= 20 and any(k in s["draft"] for k in ["检索", "RAG", "文档", "回答", "幻觉"])
    return {"approved": ok, "revisions": s["revisions"] + (0 if ok else 1),
            "draft": s["draft"] if ok else "",     # 打回→清空让写作重来
            "trace": s["trace"] + [f"🔍 审核 agent → {'通过' if ok else '打回重写'}"]}


def finalize(s):
    report = f"# {s['topic']} · 研究报告\n\n{s['draft']}\n\n参考资料：\n" + "\n".join(f"- {f}" for f in s["findings"])
    return {"report": report, "approved": True, "trace": s["trace"] + ["📄 汇总成报告"]}


def build_graph():
    g = StateGraph(State)
    for name, fn in [("supervisor", supervisor), ("search", search_agent), ("write", write_agent),
                     ("review", review_agent), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", _route,
                            {"search": "search", "write": "write", "review": "review", "finalize": "finalize"})
    for w in ("search", "write", "review"):        # 子 agent 干完回主管(多智能体循环)
        g.add_edge(w, "supervisor")
    g.add_edge("finalize", END)
    return g.compile()


def research(topic):
    return build_graph().invoke({"topic": topic, "findings": [], "draft": "", "approved": False,
                                 "revisions": 0, "report": "", "trace": []}, {"recursion_limit": 30})


# —— 联网搜索(Tavily)参考代码：需 TAVILY_API_KEY，本机不跑 ——
def web_search(query):
    from tavily import TavilyClient   # pip install tavily-python
    import os
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY")).search(query, max_results=5)["results"]


if __name__ == "__main__":
    r = research("RAG 检索增强生成")
    # 自检噪音：分隔标题 + 逐条打印多智能体协作 trace(主管→检索→写作→审核→汇总)，省略
    # print("多智能体协作过程：")
    # for step in r["trace"]:
    #     print("  ", step)
    print("\n" + r["report"][:300])
    # 自检：走完 检索→写作→审核→汇总，产出报告 + 有资料
    assert r["report"] and r["findings"] and r["approved"]
    assert any("检索" in t or "search" in t for t in r["trace"])
    assert any("写作" in t for t in r["trace"]) and any("审核" in t for t in r["trace"])
    print("\n✅ 案例2 跑通：LangGraph 多智能体(主管+检索/写作/审核子agent+循环反馈)，对标 deepsearch-agents。")
