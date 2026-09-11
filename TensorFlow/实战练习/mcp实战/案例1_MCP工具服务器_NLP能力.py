"""
================================================================================
 MCP 实战 · 案例1 · 把 NLP 能力(情感 Ch1 + NER Ch6/7)暴露成 MCP 工具
================================================================================
 为什么用 MCP：一个 AI Agent(Host，如 Claude Desktop)本身不会跑 transformers 模型。
 我们把「情感分析」「命名实体识别」两个模型能力包成 MCP 服务器暴露的 Tools，
 Agent 就能【动态发现】(tools/list) 并【按需调用】(tools/call)——模型部署在哪、
 怎么加载，Agent 完全不用管，只认 JSON-RPC 协议。这就是 MCP 的价值：能力即插即用。

 手写协议(本机没装 mcp/fastmcp SDK)：纯 Python stdio + JSON-RPC 2.0，照 chapter2 写。
   服务器 = 逐行读 stdin 的 JSON-RPC 请求、把响应写回 stdout(必须 flush)。
   客户端 = 用 subprocess 把服务器当子进程 spawn，走 initialize→tools/list→tools/call。

 用法：
   python3 案例1_MCP工具服务器_NLP能力.py            # 跑客户端(自动 spawn 服务器)，走完整流程
   python3 案例1_MCP工具服务器_NLP能力.py --server    # 只作为 MCP 服务器(供别的 Host 连)

 [Ch1 情感] distilbert-base-uncased-finetuned-sst-2-english
 [Ch6/7 NER] huggingface-course/bert-finetuned-ner (aggregation_strategy=simple 合并子词)
================================================================================
"""
import json
import subprocess
import sys


# ==============================================================================
# 一、模型层：惰性加载(生产关键)——服务器进程启动不加载，第一次 tools/call 才加载并缓存
# ==============================================================================
# 为什么惰性：MCP 服务器可能被 Host 频繁启停/探活。启动就加载模型会让握手很慢、浪费显存。
# 惰性加载 = 只有真被调用的能力才占资源；且加载一次全程复用(单例)。
_STATE = {"dev": None, "sent_tok": None, "sent": None, "ner": None}


def _pick_device():
    import torch
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def _ensure_sentiment():
    if _STATE["sent"] is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _STATE["dev"] = _STATE["dev"] or _pick_device()
        name = "distilbert-base-uncased-finetuned-sst-2-english"
        _STATE["sent_tok"] = AutoTokenizer.from_pretrained(name)
        _STATE["sent"] = AutoModelForSequenceClassification.from_pretrained(name).to(_STATE["dev"]).eval()


def _ensure_ner():
    if _STATE["ner"] is None:
        from transformers import pipeline
        _STATE["dev"] = _STATE["dev"] or _pick_device()
        _STATE["ner"] = pipeline(
            "token-classification", model="huggingface-course/bert-finetuned-ner",
            aggregation_strategy="simple",
            device=0 if _STATE["dev"] == "cuda" else -1)


def _get_sentiment(text: str):
    """[Ch1] 分词→模型 logits→softmax→argmax 取标签+概率。"""
    import torch
    _ensure_sentiment()
    enc = _STATE["sent_tok"](text, return_tensors="pt", truncation=True).to(_STATE["dev"])
    with torch.no_grad():
        probs = torch.softmax(_STATE["sent"](**enc).logits, -1)[0]
    i = int(probs.argmax())
    return {"label": _STATE["sent"].config.id2label[i], "score": round(float(probs[i]), 4)}


def _extract_entities(text: str):
    """[Ch6/7] token 分类 + aggregation_strategy=simple 把子词合并成实体级结果。"""
    _ensure_ner()
    ents = _STATE["ner"](text)
    return {"entities": [
        {"group": e["entity_group"], "word": e["word"], "score": round(float(e["score"]), 4)}
        for e in ents]}


# ==============================================================================
# 二、MCP 服务器：工具注册表(名字/描述/输入 JSON Schema 供发现) + JSON-RPC 分发
# ==============================================================================
TOOLS = {
    "get_sentiment": {
        "description": "分析英文文本的情感极性(POSITIVE/NEGATIVE)并给出置信度。",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要分析的英文文本"}},
            "required": ["text"],
        },
        "func": _get_sentiment,
    },
    "extract_entities": {
        "description": "从英文文本抽取命名实体(人名PER/机构ORG/地点LOC/其他MISC)。",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要抽实体的英文文本"}},
            "required": ["text"],
        },
        "func": _extract_entities,
    },
}


def handle_request(req: dict) -> dict:
    """MCP 服务器核心：按 JSON-RPC method 分发。"""
    method, req_id = req.get("method"), req.get("id")
    params = req.get("params", {})

    def ok(result): return {"jsonrpc": "2.0", "id": req_id, "result": result}
    def err(code, msg): return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}

    if method == "initialize":
        # 握手：声明协议版本 + 能力(这里只提供 tools)
        return ok({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "nlp-tools-server", "version": "1.0.0"},
        })

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
            # MCP 约定：结果放 content 列表；工具内部错误用 isError=True 表达(而非协议级 error)
            return ok({"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                       "isError": False})
        except Exception as e:
            return ok({"content": [{"type": "text", "text": str(e)}], "isError": True})

    return err(-32601, f"Method not found: {method}")


def run_server():
    """作为 MCP 服务器运行：逐行读 stdin，响应写回 stdout 并 flush。"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        sys.stdout.write(json.dumps(handle_request(req), ensure_ascii=False) + "\n")
        sys.stdout.flush()   # stdio 传输必须 flush，否则客户端阻塞读不到


# ==============================================================================
# 三、MCP 客户端：spawn 服务器子进程，走完整流程
# ==============================================================================
class MCPClient:
    def __init__(self, server_cmd):
        self.proc = subprocess.Popen(
            server_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
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
    # print("=" * 72 + "\n MCP 案例1：把 NLP 能力(情感 Ch1 + NER Ch6/7)暴露成工具\n" + "=" * 72)
    client = MCPClient([sys.executable, __file__, "--server"])
    try:
        info = client.call("initialize", {"protocolVersion": "2024-11-05",
                                          "capabilities": {}, "clientInfo": {"name": "nlp-client"}})
        # print(f"\n① initialize → 服务器={info['serverInfo']['name']} "
        #       f"v{info['serverInfo']['version']}，能力={list(info['capabilities'])}")

        tools = client.call("tools/list")["tools"]
        # print(f"\n② tools/list 动态发现 {len(tools)} 个工具：")   # 自检: 发现 get_sentiment / extract_entities
        # for t in tools:
        #     print(f"   · {t['name']}: {t['description']}")
        #     print(f"     输入: {list(t['inputSchema']['properties'])}")

        # print("\n③ tools/call 真实调用(第一次会惰性加载模型，稍等)：")
        s = "This new update is absolutely fantastic, I love it!"
        r1 = client.call("tools/call", {"name": "get_sentiment", "arguments": {"text": s}})
        # print(f"   get_sentiment(\"{s}\")\n     → {r1['content'][0]['text']}")   # -> label=POSITIVE

        n = "Sylvain works at Hugging Face in New York."
        r2 = client.call("tools/call", {"name": "extract_entities", "arguments": {"text": n}})
        # print(f"   extract_entities(\"{n}\")\n     → {r2['content'][0]['text']}")   # -> PER:Sylvain ORG:Hugging Face LOC:New York

        try:
            client.call("tools/call", {"name": "no_such_tool", "arguments": {}})
        except RuntimeError as e:
            print(f"\n④ 调用不存在工具 → 服务器正确报错: {e}")

        assert json.loads(r1["content"][0]["text"])["label"] == "POSITIVE"
        assert any(e["group"] == "PER" for e in json.loads(r2["content"][0]["text"])["entities"])
        print("\n✅ 自检通过：MCP(stdio+JSON-RPC) 把两个真实模型能力暴露成工具，握手/发现/调用/报错全对。")
    finally:
        client.close()


# ==============================================================================
# 生产要点(上云怎么搭)
# ==============================================================================
# · 传输：本地/桌面用 stdio(子进程)；服务化用 Streamable HTTP + SSE(见案例5)，多个 Host 可共享同一服务器。
# · 惰性加载 + 单例：模型只在首次调用时加载并常驻；高并发时前面挂 batch 队列 + 多副本。
# · 能力隔离：不同团队的模型各起一个 MCP 服务器，Host 同时连多个(NLP服务器/检索服务器/LLM服务器)。
# · 输入校验：inputSchema 是给 Host 看的契约；服务器内仍要防御式校验，超长文本截断(truncation=True)。
# · 可观测：每次 tools/call 记录 name/耗时/isError，接 Prometheus，监控各工具 QPS 与错误率。

# ==============================================================================
# 面试题(MCP 会被问什么)
# ==============================================================================
# Q: MCP 是什么？一句话。
# A: Model Context Protocol，Anthropic 开源的开放标准，用统一的 JSON-RPC 协议让 AI 应用(Host)
#    连接外部工具/数据源(Server)。类比"AI 世界的 USB-C"——一次实现，处处可插。
# Q: MCP 的四大原语(primitives)是什么？
# A: Tools(可被模型调用的函数/动作，如本例)、Resources(可读的数据/上下文，如文件)、
#    Prompts(预置的提示模板)、Sampling(服务器反过来请求 Host 的 LLM 生成)。见案例5演示前三个。
# Q: tools/list 为什么重要？和写死的 API 调用有何不同？
# A: 动态发现——Host 运行时才问服务器"你有哪些工具、参数是什么(inputSchema)"，工具可热插拔、
#    版本升级不用改 Host 代码。写死 API 则强耦合，加个能力就要改客户端。
# Q: 惰性加载模型对 MCP 服务器为什么重要？
# A: Host 会频繁探活/启停服务器，启动即加载大模型会拖慢握手、常驻占显存；惰性加载让只有被真正
#    调用的能力才占资源，握手秒回。

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_server()
    else:
        run_client()
