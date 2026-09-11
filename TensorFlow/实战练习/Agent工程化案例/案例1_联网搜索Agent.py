"""
================================================================================
 Agent 工程化案例1 · 联网搜索 Agent（DuckDuckGo 真联网 + LangGraph + mlx，本机真跑）
================================================================================
 补齐"联网搜索/深度研究"短板：Agent 真的去搜索引擎查最新资料 → 汇总成带来源的回答。
 技术：DuckDuckGo 真联网搜索(免 key，本机可跑) + LangGraph 编排[搜索→读取→mlx 综合] + mlx-lm 生成。
 (生产也可换 Tavily：TavilyClient(api_key).search(...)，见文末参考；DuckDuckGo 无需 key 更适合本机演示。)
 跑：python3 案例1_联网搜索Agent.py "你的问题"   （需联网）
================================================================================
"""
import sys
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

_M = {}


def _ddg(query, k=4):
    """真联网搜索(DuckDuckGo，免 key)。"""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    with DDGS() as d:
        return [{"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
                for r in d.text(query, max_results=k)]


def _tavily(query, k=4):
    """真联网搜索(Tavily，为 LLM 优化，需 TAVILY_API_KEY)。真实代码，有 key 即可跑。"""
    import os
    from tavily import TavilyClient
    res = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")).search(query, max_results=k)
    return [{"title": r.get("title", ""), "body": r.get("content", ""), "href": r.get("url", "")}
            for r in res["results"]]


def _web_search(query, k=4):
    """搜索后端可选：有 TAVILY_API_KEY 用 Tavily(生产常用)，否则 DuckDuckGo(免 key，本机演示)。"""
    import os
    return _tavily(query, k) if os.getenv("TAVILY_API_KEY") else _ddg(query, k)


def _mlx(prompt, n=160):
    if "m" not in _M:
        from mlx_lm import load
        _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import generate
    t = _M["t"].apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    return generate(_M["m"], _M["t"], prompt=t, max_tokens=n, verbose=False).strip()


class State(TypedDict):
    query: str
    results: list
    answer: str
    sources: list
    trace: list


def n_search(s):
    import os
    backend = "Tavily" if os.getenv("TAVILY_API_KEY") else "DuckDuckGo"
    res = _web_search(s["query"], k=4)
    return {"results": res, "sources": [r["href"] for r in res],
            "trace": s["trace"] + [f"🔎 联网搜索({backend}) → {len(res)} 条结果"]}


def n_read(s):
    # 读取/裁剪搜索片段(生产可再抓正文 VisitWebpage；这里用摘要片段)
    return {"trace": s["trace"] + [f"📄 读取 {len(s['results'])} 条搜索片段"]}


def n_synthesize(s):
    ctx = "\n".join(f"[{i+1}] {r['title']}：{r['body'][:160]}" for i, r in enumerate(s["results"]))
    ans = _mlx(f"根据下面联网搜到的资料，用 2-3 句话回答问题，并在句末标引用编号。\n"
               f"资料：\n{ctx}\n\n问题：{s['query']}\n回答：", 160)
    return {"answer": ans, "trace": s["trace"] + ["🧠 mlx 综合成带来源的回答"]}


def build_graph():
    g = StateGraph(State)
    g.add_node("search", n_search)
    g.add_node("read", n_read)
    g.add_node("synthesize", n_synthesize)
    g.add_edge(START, "search")
    g.add_edge("search", "read")
    g.add_edge("read", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def research(query):
    return build_graph().invoke({"query": query, "results": [], "answer": "", "sources": [], "trace": []})


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "什么是 RAG 检索增强生成"
    r = research(q)
    print(f"❓ {q}")
    # 自检噪音：逐条打印 trace 过程步(🔎搜索→📄读取→🧠综合)，省略以减少输出
    # for step in r["trace"]:
    #     print("  ", step)
    print("\n💬 回答：", r["answer"][:220])
    # print("\n📎 来源：")
    for u in r["sources"][:4]:
        print("   -", u)
    assert r["results"] and r["answer"], "联网搜索或生成失败(检查网络)"
    print("\n✅ 案例1 跑通：DuckDuckGo 真联网搜索 + LangGraph 编排 + mlx 综合带来源回答。")
