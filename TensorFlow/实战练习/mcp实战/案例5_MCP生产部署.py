"""
================================================================================
 MCP 实战 · 案例5 · MCP 生产化：三大原语(Tools/Resources/Prompts) + 部署全景
================================================================================
 前四个案例聚焦 Tools。真实 MCP 服务器还提供另两个原语，本案例用一个完整服务器演示三者：
   · Tools     可被模型调用的"动作/函数"(有副作用/计算)。这里: analyze_text(情感+词数)。
   · Resources 可被读取的"数据/上下文"(用 URI 寻址，只读)。这里: docs://faq、docs://models。
   · Prompts   预置的"提示模板"(供 Host 让用户一键选用)。这里: summarize / triage 模板。
 客户端演示完整发现+读取: tools/list、resources/list+resources/read、prompts/list+prompts/get。

 —— 生产部署要点全在文件下半部注释：stdio vs HTTP+SSE 传输、多服务器、鉴权、
    Claude Desktop 集成(claude_desktop_config.json)、官方 FastMCP 等价写法。

 手写协议(本机无 SDK)：纯 Python stdio + JSON-RPC 2.0。
 用法：
   python3 案例5_MCP生产部署.py            # 客户端(自动 spawn 服务器)，演示三大原语
   python3 案例5_MCP生产部署.py --server    # 只作为 MCP 服务器
================================================================================
"""
import json
import subprocess
import sys


# ==============================================================================
# 一、三大原语的数据/实现
# ==============================================================================
# --- Resources：只读数据，用 URI 寻址（生产里可以是文件/数据库/API 快照）---
RESOURCES = {
    "docs://faq": {
        "name": "faq",
        "mimeType": "text/plain",
        "description": "常见问题解答全文(供模型作为上下文阅读)。",
        "text": ("Q: How to reset password? A: Settings > Security > Reset Password.\n"
                 "Q: Refund for double charge? A: Auto-refunded in 3-5 business days.\n"
                 "Q: Free storage? A: 5GB free, 1TB on Pro."),
    },
    "docs://models": {
        "name": "models",
        "mimeType": "application/json",
        "description": "本服务可用的模型清单(元数据)。",
        "text": json.dumps({"sentiment": "distilbert-sst2", "ner": "bert-finetuned-ner",
                            "embed": "all-MiniLM-L6-v2", "llm": "SmolLM2-135M-Instruct"}),
    },
}

# --- Prompts：预置提示模板，带参数占位；Host 让用户选模板、填参数 ---
PROMPTS = {
    "summarize": {
        "description": "把一段文本总结成一句话。",
        "arguments": [{"name": "text", "description": "要总结的文本", "required": True}],
        "template": "Summarize the following text in one concise sentence:\n\n{text}",
    },
    "triage": {
        "description": "客服工单分诊提示：判断紧急度并给出处理建议。",
        "arguments": [{"name": "ticket", "description": "工单内容", "required": True}],
        "template": ("You are a support triage assistant. Classify the urgency (low/medium/high) "
                     "of this ticket and suggest the next action:\n\nTicket: {ticket}"),
    },
}


# --- Tools：动作。这里 analyze_text 惰性加载情感模型(复用 Ch1 能力) ---
_STATE = {"dev": None, "tok": None, "sent": None}


def _pick_device():
    import torch
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def _analyze_text(text: str):
    import torch
    if _STATE["sent"] is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _STATE["dev"] = _pick_device()
        m = "distilbert-base-uncased-finetuned-sst-2-english"
        _STATE["tok"] = AutoTokenizer.from_pretrained(m)
        _STATE["sent"] = AutoModelForSequenceClassification.from_pretrained(m).to(_STATE["dev"]).eval()
    enc = _STATE["tok"](text, return_tensors="pt", truncation=True).to(_STATE["dev"])
    with torch.no_grad():
        p = torch.softmax(_STATE["sent"](**enc).logits, -1)[0]
    i = int(p.argmax())
    return {"sentiment": _STATE["sent"].config.id2label[i], "score": round(float(p[i]), 4),
            "word_count": len(text.split())}


TOOLS = {
    "analyze_text": {
        "description": "分析文本：返回情感极性、置信度和词数。",
        "inputSchema": {"type": "object",
                        "properties": {"text": {"type": "string"}}, "required": ["text"]},
        "func": _analyze_text,
    },
}


# ==============================================================================
# 二、MCP 服务器：在 initialize 里声明三种能力，并实现各自的 list/read/get/call
# ==============================================================================
def handle_request(req: dict) -> dict:
    method, req_id = req.get("method"), req.get("id")
    params = req.get("params", {})

    def ok(result): return {"jsonrpc": "2.0", "id": req_id, "result": result}
    def err(code, msg): return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}

    if method == "initialize":
        # 三大原语都声明在 capabilities 里，Host 才知道能问 resources/prompts
        return ok({"protocolVersion": "2024-11-05",
                   "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                   "serverInfo": {"name": "production-demo-server", "version": "1.0.0"}})

    # --- Tools ---
    if method == "tools/list":
        return ok({"tools": [{"name": n, "description": t["description"],
                              "inputSchema": t["inputSchema"]} for n, t in TOOLS.items()]})
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

    # --- Resources ---
    if method == "resources/list":
        return ok({"resources": [{"uri": u, "name": r["name"], "mimeType": r["mimeType"],
                                  "description": r["description"]} for u, r in RESOURCES.items()]})
    if method == "resources/read":
        uri = params.get("uri")
        if uri not in RESOURCES:
            return err(-32602, f"Unknown resource: {uri}")
        r = RESOURCES[uri]
        return ok({"contents": [{"uri": uri, "mimeType": r["mimeType"], "text": r["text"]}]})

    # --- Prompts ---
    if method == "prompts/list":
        return ok({"prompts": [{"name": n, "description": p["description"],
                               "arguments": p["arguments"]} for n, p in PROMPTS.items()]})
    if method == "prompts/get":
        name, args = params.get("name"), params.get("arguments", {})
        if name not in PROMPTS:
            return err(-32602, f"Unknown prompt: {name}")
        p = PROMPTS[name]
        try:
            filled = p["template"].format(**args)   # 用参数渲染模板
        except KeyError as e:
            return err(-32602, f"Missing prompt argument: {e}")
        return ok({"description": p["description"],
                   "messages": [{"role": "user",
                                 "content": {"type": "text", "text": filled}}]})

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
# 三、客户端：演示三大原语的完整发现/读取
# ==============================================================================
class MCPClient:
    def __init__(self, cmd):
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
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
        self.proc.stdin.close(); self.proc.terminate()


def run_client():
    # print("=" * 72 + "\n MCP 案例5：一个服务器，三大原语 Tools + Resources + Prompts\n" + "=" * 72)
    c = MCPClient([sys.executable, __file__, "--server"])
    try:
        info = c.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                     "clientInfo": {"name": "prod-client"}})
        print(f"\n① initialize → {info['serverInfo']['name']}，声明能力 = {list(info['capabilities'])}")

        # print("\n② Tools（动作）：")
        tools = c.call("tools/list")["tools"]
        print(f"   tools/list → {[t['name'] for t in tools]}")
        r = c.call("tools/call", {"name": "analyze_text",
                                  "arguments": {"text": "This service is wonderful and fast!"}})
        print(f"   tools/call analyze_text → {r['content'][0]['text']}")

        # print("\n③ Resources（只读数据，URI 寻址）：")
        res = c.call("resources/list")["resources"]
        for x in res:
            print(f"   · {x['uri']}  ({x['mimeType']})  {x['description']}")
        read = c.call("resources/read", {"uri": "docs://models"})
        print(f"   resources/read docs://models → {read['contents'][0]['text']}")

        # print("\n④ Prompts（预置提示模板）：")
        prompts = c.call("prompts/list")["prompts"]
        for x in prompts:
            argnames = [a["name"] for a in x["arguments"]]
            print(f"   · {x['name']}{argnames}: {x['description']}")
        got = c.call("prompts/get", {"name": "triage", "arguments": {"ticket": "I was double charged!"}})
        print(f"   prompts/get triage → 渲染后的提示:\n     \"{got['messages'][0]['content']['text']}\"")

        assert json.loads(r["content"][0]["text"])["sentiment"] == "POSITIVE"
        assert "SmolLM2" in read["contents"][0]["text"]
        assert "double charged" in got["messages"][0]["content"]["text"]
        print("\n✅ 自检通过：一个 MCP 服务器完整提供 Tools/Resources/Prompts 三大原语，发现与读取全对。")
    finally:
        c.close()


# ==============================================================================
# ★ 生产部署全景（这是本案例的重点，务必读注释）
# ==============================================================================
# 1) 传输(Transport)：
#    · stdio     —— 服务器=本地子进程，Host 通过其 stdin/stdout 收发(前面所有案例)。适合桌面/本地工具、
#                    Claude Desktop。零网络配置、天然隔离，但只能本机一对一。
#    · Streamable HTTP + SSE —— 服务器是个 HTTP 服务(POST 发请求，SSE 推流式响应)。适合远程/多客户端/
#                    云端；能承载流式生成、水平扩缩容、放到 K8s + 网关后。MCP 现代远程标准传输。
#    协议消息(JSON-RPC 的 initialize/tools/... )两种传输完全一样，只是"管道"不同——换传输不改业务逻辑。
#
# 2) 多服务器：一个 Host 同时连多个 MCP 服务器(案例4已演示)。生产里按领域拆：nlp-server / retrieval-server
#    / db-server / github-server 各自独立部署、独立扩缩容、独立发版；Host 侧聚合成一张能力表。
#
# 3) 鉴权与安全：
#    · 远程(HTTP)传输用 OAuth 2.1 / Bearer Token 鉴权；MCP 规范已纳入 OAuth 授权框架。
#    · 最小权限：不同用户/租户可见的工具集不同(RBAC)；敏感工具二次确认(human-in-the-loop)。
#    · 提示注入防护：工具返回的数据不可信，不能直接当指令执行；对参数做校验、对输出做隔离/审计。
#    · 隔离：MCP 服务器按容器/沙箱运行，限制文件/网络访问范围。
#
# 4) Claude Desktop 集成：在 claude_desktop_config.json 里注册本服务器(stdio)：
#    {
#      "mcpServers": {
#        "nlp-tools": {
#          "command": "python3",
#          "args": ["/绝对路径/案例5_MCP生产部署.py", "--server"]
#        }
#      }
#    }
#    重启 Claude Desktop 后，Claude 就能发现并调用这里的工具/资源/提示。远程服务器则配 URL + 鉴权头。
#
# 5) 官方 FastMCP 等价写法(本机没装，仅作对照——装了 `pip install mcp` 后几行即可，无需手写 JSON-RPC)：
#    # from mcp.server.fastmcp import FastMCP
#    # mcp = FastMCP("production-demo-server")
#    # @mcp.tool()
#    # def analyze_text(text: str) -> dict: ...          # 装饰器自动生成 inputSchema
#    # @mcp.resource("docs://faq")
#    # def faq() -> str: ...                              # 自动挂 resources/list + read
#    # @mcp.prompt()
#    # def summarize(text: str) -> str: ...               # 自动挂 prompts/list + get
#    # mcp.run(transport="stdio")   # 或 transport="streamable-http" 上远程
#    FastMCP 帮你处理握手/schema 生成/传输；手写版(本文件)让你看清它底下就是这套 JSON-RPC 消息。
#
# 6) 可观测与可靠性：结构化日志(每次调用 method/工具名/耗时/isError)、指标(QPS/延迟/错误率)、
#    超时与重试、优雅关闭、版本协商(protocolVersion)。上线前压测各工具并发。

# ==============================================================================
# 面试题(MCP 生产/原语)
# ==============================================================================
# Q: MCP 三大服务器原语(Tools/Resources/Prompts)分别是什么、怎么选？
# A: Tools=模型可调用的"动作/函数"(有计算/副作用，如查天气、跑模型)；Resources=只读"数据/上下文"(URI
#    寻址，如文件/表，供模型阅读)；Prompts=预置"提示模板"(供用户在 Host 里一键选用、填参)。要"做事"用
#    Tools，要"给料读"用 Resources，要"给现成提示"用 Prompts。(还有 Sampling：服务器反请 Host 的 LLM。)
# Q: stdio 和 HTTP+SSE 传输怎么选？
# A: 本地/桌面/单机工具用 stdio(子进程，零配置、隔离好)；远程/多客户端/云端/需流式与扩缩容用 Streamable
#    HTTP+SSE。业务消息(JSON-RPC)不变，只换管道。
# Q: 远程 MCP 服务器怎么做鉴权？
# A: MCP 规范采用 OAuth 2.1 授权框架，HTTP 传输带 Bearer Token；再叠加 RBAC(按用户暴露不同工具集)、
#    最小权限、敏感操作人工确认、沙箱隔离。
# Q: MCP 最大的安全风险是什么？
# A: 提示注入 + 过度授权：工具/资源返回的内容不可信，若被 LLM 当指令执行、或工具权限过大，可能被诱导做
#    危险操作。对策：输出隔离/不可执行、参数校验、最小权限、human-in-the-loop、全程审计。
# Q: 为什么手写和 FastMCP 说的是"同一套东西"？
# A: 两者产生的都是 initialize/tools/list/tools/call/resources/*/prompts/* 这些 JSON-RPC 消息。FastMCP 用
#    装饰器自动生成 schema 和处理传输，手写版把这些消息显式写出来——协议一致，只是抽象层级不同。

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_server()
    else:
        run_client()
