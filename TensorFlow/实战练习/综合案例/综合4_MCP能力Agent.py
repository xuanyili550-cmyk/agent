"""
================================================================================
 综合案例4 · MCP 能力 Agent（整合 MCP + Agent 编排 + 本地检索）
================================================================================
 一个 Agent 同时拥有两类工具：
   · 【远程 MCP 工具】：通过 MCP 协议连 ../mcp实战/案例1 的 NLP 服务器(情感 Ch1 + NER Ch6/7)——
     工具部署在别处、即插即用，Agent 只认协议。
   · 【本地工具】：bge 中文知识库检索。
 Agent 按任务把请求路由到 MCP 工具或本地工具，体现“Agent + MCP + RAG”的综合。
 编排用确定性规则(稳)；生产可换 smolagents ToolCollection.from_mcp + LLM(见 ../Agent实战/案例4)。
 跑：python3 综合4_MCP能力Agent.py
================================================================================
"""
import os
import re
import sys
import json
import subprocess

MCP_SERVER = os.path.join(os.path.dirname(__file__), "..", "mcp实战",
                          "案例1_MCP工具服务器_NLP能力.py")
_M = {}
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
KB = ["退货政策：7 天无理由退货，商品需保持完好。",
      "配送时效：同城次日达，跨省 3-5 天。",
      "会员权益：付费会员享 9 折 + 优先客服。"]


# ---- 远程能力：手写 MCP stdio 客户端 ----
class MCP:
    def __init__(self, path):
        self.p = subprocess.Popen([sys.executable, path, "--server"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.i = 0

    def call(self, method, params=None):
        self.i += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.i, "method": method,
                                       "params": params or {}}, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())["result"]

    def tool(self, name, **args):
        return self.call("tools/call", {"name": name, "arguments": args})["content"][0]["text"]

    def close(self):
        self.p.stdin.close(); self.p.terminate()


# ---- 本地能力：bge 检索 ----
def _search_kb(q):
    import torch
    import torch.nn.functional as F
    if "emb" not in _M:
        from transformers import AutoTokenizer, AutoModel
        n = "BAAI/bge-small-zh-v1.5"
        d = "mps" if torch.backends.mps.is_available() else "cpu"
        _M["d"], _M["et"] = d, AutoTokenizer.from_pretrained(n)
        _M["em"] = AutoModel.from_pretrained(n).to(d).eval()

    def emb(ts, query=False):
        if query:
            ts = [QUERY_PREFIX + t for t in ts]
        enc = _M["et"](ts, padding=True, truncation=True, return_tensors="pt").to(_M["d"])
        with torch.no_grad():
            v = _M["em"](**enc).last_hidden_state[:, 0]
        return F.normalize(v, p=2, dim=1)
    if "kv" not in _M:
        _M["kv"] = emb(KB)
    sims = (emb([q], query=True) @ _M["kv"].T)[0]
    return KB[int(sims.argmax())]


# ---- Agent 编排：按任务路由到 MCP 工具或本地检索 ----
def agent(mcp, task):
    kind, payload = task["type"], task["text"]
    if kind == "sentiment":                              # → 远程 MCP 工具(英文情感)
        return "MCP.get_sentiment", mcp.tool("get_sentiment", text=payload)
    if kind == "entities":                               # → 远程 MCP 工具(英文 NER)
        return "MCP.extract_entities", mcp.tool("extract_entities", text=payload)
    return "local.search_kb", _search_kb(payload)        # → 本地检索(中文知识库)


if __name__ == "__main__":
    mcp = MCP(MCP_SERVER)
    try:
        tools = [t["name"] for t in mcp.call("tools/list")["tools"]]
        print(f">>> 已连 MCP 服务器，发现远程工具：{tools} + 本地工具：search_kb\n")
        tasks = [
            {"type": "sentiment", "text": "This product is absolutely fantastic!"},
            {"type": "entities", "text": "Tim Cook works at Apple in Cupertino."},
            {"type": "kb", "text": "退货政策是怎样的？"},
        ]
        for t in tasks:
            tool, obs = agent(mcp, t)
            # print("─" * 62)  # 装饰分隔线（静音）
            print(f"任务[{t['type']}]: {t['text']}\n  🔧 Agent 调用 {tool}\n  👀 {obs}")
        # 自检：MCP 工具 + 本地检索都通
        assert json.loads(mcp.tool("get_sentiment", text="I love it!"))["label"] == "POSITIVE"
        assert "退货" in _search_kb("怎么退货")
        print("\n✅ 综合4 跑通：Agent 同时用【远程 MCP 工具(情感/NER)】+【本地检索】，按任务路由编排。")
    finally:
        mcp.close()
