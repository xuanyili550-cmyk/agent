"""
================================================================================
 MCP 实战 · 案例2 · 把 RAG 语义检索(Ch5/6)暴露成 MCP 工具 search_knowledge
================================================================================
 为什么用 MCP 包检索：Agent/LLM 本身不知道你的私有知识库。把「按语义检索知识库」做成
 一个 MCP 工具，Agent 在回答前先 tools/call("search_knowledge") 取回最相关的资料，再据此
 回答——这正是 RAG(检索增强生成)的"检索"环节，只是通过标准 MCP 协议暴露，任何 Host 可复用。

 检索原理(Ch5/6)：
   ① 把知识库每条文本用句向量模型嵌成向量(mask 加权 mean 池化 + L2 归一化)。
   ② 查询也嵌成向量，与库里所有向量算余弦相似度(归一化后 = 点积)。
   ③ 取 Top-K 返回。换个说法也能命中(语义匹配)，胜过关键词。

 手写 MCP：纯 Python stdio + JSON-RPC 2.0(照 chapter2)。模型惰性加载。
 用法：
   python3 案例2_MCP服务器_语义检索.py            # 客户端(自动 spawn 服务器)走完整流程
   python3 案例2_MCP服务器_语义检索.py --server    # 只作为 MCP 服务器

 [Ch5/6] sentence-transformers/all-MiniLM-L6-v2
================================================================================
"""
import json
import subprocess
import sys


# ==============================================================================
# 一、小知识库(硬编码演示；生产放向量数据库 Qdrant/Milvus)
# ==============================================================================
KNOWLEDGE = [
    "To reset your password, go to Settings > Security > Reset Password and follow the email link.",
    "The photo-upload crash was fixed in app version 3.2. Please upgrade to v3.2 or later.",
    "Duplicate charges are automatically refunded within 3-5 business days after verification.",
    "Free accounts include 5GB of storage; upgrade to the Pro plan for 1TB of storage.",
    "Our support hotline is open from 9am to 6pm on weekdays; type 'agent' in chat for a human.",
    "You can export your data as CSV or JSON from the Account > Data Export page at any time.",
]

# 模型状态(惰性加载 + 单例)：库向量只算一次并缓存
_STATE = {"dev": None, "tok": None, "model": None, "kb_vecs": None}


def _pick_device():
    import torch
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def _embed(texts):
    """[Ch5] mask 加权 mean 池化 + L2 归一化 → 单位向量，点积即余弦。"""
    import torch
    import torch.nn.functional as F
    enc = _STATE["tok"](texts, padding=True, truncation=True, return_tensors="pt").to(_STATE["dev"])
    with torch.no_grad():
        out = _STATE["model"](**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    v = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(v, p=2, dim=1)


def _ensure_model():
    if _STATE["model"] is None:
        from transformers import AutoTokenizer, AutoModel
        _STATE["dev"] = _pick_device()
        name = "sentence-transformers/all-MiniLM-L6-v2"
        _STATE["tok"] = AutoTokenizer.from_pretrained(name)
        _STATE["model"] = AutoModel.from_pretrained(name).to(_STATE["dev"]).eval()
        _STATE["kb_vecs"] = _embed(KNOWLEDGE)   # 库向量预计算并缓存(生产=写进向量库)


def _search_knowledge(query: str, top_k: int = 3):
    """语义检索工具：返回 Top-K 最相关知识条目 + 余弦得分。"""
    _ensure_model()
    top_k = max(1, min(int(top_k), len(KNOWLEDGE)))
    sims = (_embed([query]) @ _STATE["kb_vecs"].T)[0]   # [Ch5] 余弦相似度
    order = sims.argsort(descending=True)[:top_k]
    return {"query": query, "results": [
        {"rank": r + 1, "score": round(float(sims[i]), 4), "text": KNOWLEDGE[int(i)]}
        for r, i in enumerate(order)]}


# ==============================================================================
# 二、MCP 服务器
# ==============================================================================
TOOLS = {
    "search_knowledge": {
        "description": "在知识库里按语义(非关键词)检索，返回最相关的 Top-K 条目及相似度得分。RAG 检索环节。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户的自然语言问题"},
                "top_k": {"type": "integer", "description": "返回条数，默认3", "default": 3},
            },
            "required": ["query"],
        },
        "func": _search_knowledge,
    },
}


def handle_request(req: dict) -> dict:
    method, req_id = req.get("method"), req.get("id")
    params = req.get("params", {})

    def ok(result): return {"jsonrpc": "2.0", "id": req_id, "result": result}
    def err(code, msg): return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}

    if method == "initialize":
        return ok({"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                   "serverInfo": {"name": "retrieval-server", "version": "1.0.0"}})
    if method == "tools/list":
        return ok({"tools": [
            {"name": n, "description": t["description"], "inputSchema": t["inputSchema"]}
            for n, t in TOOLS.items()]})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments", {})
        if name not in TOOLS:
            return err(-32602, f"Unknown tool: {name}")
        try:
            result = TOOLS[name]["func"](**args)
            return ok({"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                       "isError": False})
        except Exception as e:
            return ok({"content": [{"type": "text", "text": str(e)}], "isError": True})
    return err(-32601, f"Method not found: {method}")


def run_server():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        sys.stdout.write(json.dumps(handle_request(req), ensure_ascii=False) + "\n")
        sys.stdout.flush()


# ==============================================================================
# 三、MCP 客户端
# ==============================================================================
class MCPClient:
    def __init__(self, server_cmd):
        self.proc = subprocess.Popen(server_cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline())
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        return resp["result"]

    def close(self):
        self.proc.stdin.close()
        self.proc.terminate()


def run_client():
    # print("=" * 72 + "\n MCP 案例2：把 RAG 语义检索(Ch5/6)暴露成工具 search_knowledge\n" + "=" * 72)
    client = MCPClient([sys.executable, __file__, "--server"])
    try:
        info = client.call("initialize", {"protocolVersion": "2024-11-05",
                                          "capabilities": {}, "clientInfo": {"name": "rag-client"}})
        # print(f"\n① initialize → 服务器={info['serverInfo']['name']} v{info['serverInfo']['version']}")

        tools = client.call("tools/list")["tools"]
        # print(f"\n② tools/list 发现 {len(tools)} 个工具：{[t['name'] for t in tools]}")   # 自检: 发现 search_knowledge

        # print("\n③ tools/call 语义检索(注意：查询用词和知识库不同，靠语义命中)：")
        queries = [
            "I forgot my login credentials, how do I recover access?",  # ≠ "reset password" 字面
            "The app keeps crashing when I add pictures",               # ≠ "photo-upload" 字面
            "how much space do I get for free?",                        # ≠ "storage" 字面
        ]
        first = None
        for q in queries:
            r = client.call("tools/call", {"name": "search_knowledge",
                                           "arguments": {"query": q, "top_k": 2}})
            data = json.loads(r["content"][0]["text"])
            first = first or data
            # print(f"\n   Q: {q}")
            # for hit in data["results"]:   # 自检: Top-K 命中片段(换说法也命中, 如 password/photo/storage)
            #     print(f"     [{hit['rank']}] {hit['score']:.3f}  {hit['text']}")

        top = first["results"][0]
        assert "password" in top["text"].lower() and top["score"] > 0.4
        print("\n✅ 自检通过：MCP 暴露的语义检索能换说法命中正确知识条目(Top-1 得分>0.4)——RAG 检索环节跑通。")
    finally:
        client.close()


# ==============================================================================
# 生产要点(上云怎么搭)
# ==============================================================================
# · 向量库：知识条数上千/上万就别用内存矩阵，换 Qdrant/Milvus/pgvector，支持增量写入 + ANN 近似检索。
# · 分块(chunking)：长文档先切块(按段/token 窗口 + 重叠)再嵌入，检索粒度更细；元数据存来源用于引用。
# · 混合检索：语义(向量)+ 关键词(BM25)融合(RRF)召回更全；再上 rerank 交叉编码器精排。
# · 嵌入一致性：入库和查询必须用同一个嵌入模型/同样池化方式，换模型要全量重建索引。
# · MCP 层：把 search_knowledge 做成独立检索服务器，多个业务 Host 共享；返回带 source 便于 LLM 引用溯源。

# ==============================================================================
# 面试题(MCP + RAG)
# ==============================================================================
# Q: 为什么把 RAG 检索包成 MCP 工具，而不是直接写进 Agent 代码？
# A: 解耦与复用——检索服务(向量库、嵌入模型、分块策略)独立演进，多个 Agent/Host 通过标准协议共享同一个
#    检索能力；Agent 只管"调 search_knowledge"，不关心底层换没换向量库。
# Q: MCP Tools 和 Resources 用哪个装知识库更合适？
# A: 看语义。Resources 是"可被读取的上下文/数据"(如把某文档直接暴露给模型读)；这里是"带参数的检索动作"
#    (输入 query 返回 Top-K)，属于 Tools。静态整份文档用 Resources，动态查询用 Tools。
# Q: 语义检索为什么用向量+余弦而不是关键词？
# A: 关键词要字面命中("reset password" vs "forgot credentials"就漏)；向量把语义压进空间，近义/换说法也
#    靠得近，余弦度量方向相似度、与长度无关。归一化后余弦=点积，算得快。
# Q: MCP 工具返回结果太长(整篇文档)会有什么问题？
# A: 撑爆 Host 的上下文窗口、增加 token 成本。应只回 Top-K 精简片段 + 来源，让 LLM 基于片段作答。

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_server()
    else:
        run_client()
