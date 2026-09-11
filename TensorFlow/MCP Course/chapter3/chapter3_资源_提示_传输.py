"""
================================================================================
 MCP Course · Chapter 3 · 完整原语：资源(Resources) + 提示(Prompts) + 传输层
================================================================================
 Chapter 2 实现了工具(Tools)。本章把 MCP 的四大原语补全，仍是真实 JSON-RPC 协议：
   · Resources 资源  ：只读数据源(如文件/数据库)。resources/list 发现、resources/read 读取。
   · Prompts   提示  ：预定义模板/工作流。prompts/list 发现、prompts/get 取渲染后的消息。
   · Sampling  采样  ：服务器反过来请求客户端跑 LLM(递归)。这里说明其消息形态。
   · Transport 传输  ：stdio(本地子进程) vs HTTP/SSE(远程)，两者跑同一套 JSON-RPC。

 用法：
   python3 chapter3_资源_提示_传输.py            # 客户端：发现并使用 资源 + 提示
   python3 chapter3_资源_提示_传输.py --server    # 作为 MCP 服务器运行

 生产对照：官方 SDK(FastMCP)里这些就是 @mcp.resource() / @mcp.prompt() 装饰器(见 chapter4)。
================================================================================
"""

import json
import subprocess
import sys

# ---- 资源：只读数据源(用 URI 标识，像 REST 的资源) ----
RESOURCES = {
    "docs://readme": {"name": "README", "mimeType": "text/markdown",
                      "text": "# Demo Project\nThis is a sample MCP resource."},
    "config://settings": {"name": "Settings", "mimeType": "application/json",
                          "text": json.dumps({"theme": "dark", "lang": "zh"})},
}

# ---- 提示：预定义模板，prompts/get 时把参数填进去返回一组 messages ----
def _code_review_prompt(code="", language="python"):
    return [
        {"role": "system", "content": f"You are a {language} code reviewer. "
                                      "Highlight best practices, issues, and improvements."},
        {"role": "user", "content": f"Review this {language} code:\n```{language}\n{code}\n```"},
    ]

PROMPTS = {
    "code_review": {
        "description": "Generate a code review for a snippet.",
        "arguments": [
            {"name": "code", "description": "Code to review", "required": True},
            {"name": "language", "description": "Programming language", "required": False},
        ],
        "func": _code_review_prompt,
    },
}


def handle_request(req):
    method, rid, params = req.get("method"), req.get("id"), req.get("params", {})
    ok = lambda r: {"jsonrpc": "2.0", "id": rid, "result": r}
    err = lambda c, m: {"jsonrpc": "2.0", "id": rid, "error": {"code": c, "message": m}}

    if method == "initialize":
        return ok({"protocolVersion": "2024-11-05",
                   "capabilities": {"resources": {}, "prompts": {}},  # 声明支持 资源+提示
                   "serverInfo": {"name": "demo-full-server", "version": "1.0.0"}})

    # ---- 资源 ----
    if method == "resources/list":
        return ok({"resources": [
            {"uri": u, "name": r["name"], "mimeType": r["mimeType"]}
            for u, r in RESOURCES.items()]})
    if method == "resources/read":
        uri = params.get("uri")
        if uri not in RESOURCES:
            return err(-32602, f"Resource not found: {uri}")
        r = RESOURCES[uri]
        return ok({"contents": [{"uri": uri, "mimeType": r["mimeType"], "text": r["text"]}]})

    # ---- 提示 ----
    if method == "prompts/list":
        return ok({"prompts": [
            {"name": n, "description": p["description"], "arguments": p["arguments"]}
            for n, p in PROMPTS.items()]})
    if method == "prompts/get":
        name = params.get("name")
        if name not in PROMPTS:
            return err(-32602, f"Prompt not found: {name}")
        args = params.get("arguments", {})
        msgs = PROMPTS[name]["func"](**args)
        # MCP prompts/get 返回渲染好的 messages(客户端可直接喂给 LLM)
        return ok({"description": PROMPTS[name]["description"],
                   "messages": [{"role": m["role"],
                                 "content": {"type": "text", "text": m["content"]}} for m in msgs]})

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
        sys.stdout.write(json.dumps(handle_request(req)) + "\n")
        sys.stdout.flush()


class MCPClient:
    def __init__(self, cmd):
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        self.proc.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline())
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        return resp["result"]

    def close(self):
        self.proc.stdin.close(); self.proc.terminate()


def run_client():
    print("=" * 72 + "\n MCP 客户端：发现并使用 资源(Resources) + 提示(Prompts)\n" + "=" * 72)
    c = MCPClient([sys.executable, __file__, "--server"])
    try:
        info = c.call("initialize", {})
        print(f"\n握手成功，服务器能力 = {list(info['capabilities'])}")

        # 资源：发现 → 读取
        res = c.call("resources/list")["resources"]
        print(f"\n① resources/list 发现 {len(res)} 个资源：")
        for r in res:
            print(f"   · {r['uri']} ({r['mimeType']}) — {r['name']}")
        content = c.call("resources/read", {"uri": "config://settings"})["contents"][0]
        print("   resources/read config://settings →", content["text"])

        # 提示：发现 → 渲染
        prompts = c.call("prompts/list")["prompts"]
        print(f"\n② prompts/list 发现 {len(prompts)} 个提示：")
        for p in prompts:
            print(f"   · {p['name']}: {p['description']}  参数={[a['name'] for a in p['arguments']]}")
        got = c.call("prompts/get", {"name": "code_review",
                                     "arguments": {"code": "def f(): return 1", "language": "python"}})
        print("   prompts/get code_review → 渲染出", len(got["messages"]), "条 messages：")
        for m in got["messages"]:
            print(f"     [{m['role']}] {m['content']['text'][:60]}...")

        assert content["text"] and len(got["messages"]) == 2
        print("\n✅ 自检通过：资源(发现/读取) + 提示(发现/渲染) 全走通，都是真 JSON-RPC。")
    finally:
        c.close()

    # 采样(Sampling) + 传输(Transport)：概念 + 消息形态
    print("""
③ 采样 Sampling(服务器反向请求客户端跑 LLM)：
   服务器发 sampling/createMessage 请求，params 里带 messages + maxTokens，
   客户端(Host)拿去调 LLM，把结果回给服务器 → 实现“服务器让 AI 帮它再想一步”的递归。
   例：写作 App 的服务器把草稿发给客户端 LLM，要它自我评审再改进。

④ 传输 Transport(同一套 JSON-RPC，换管道)：
   · stdio      : 服务器是本地子进程，走 stdin/stdout(本章 chapter2/3 用的就是它)。适合本地工具。
   · HTTP+SSE   : 服务器是远程 HTTP 服务，请求走 POST、服务器推送走 SSE。适合远程/多客户端。
   换传输不改业务逻辑——工具/资源/提示的 JSON-RPC 消息完全一样。
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_server()
    else:
        run_client()
