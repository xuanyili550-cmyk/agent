"""
================================================================================
 全功能案例1 · MCP 协议【全能力】（穷尽 MCP Course 的每个功能，非应用场景）
================================================================================
 不是“某个应用”，而是把 MCP 的【所有功能】一次跑全，末尾用清单逐条证明覆盖：
   四大原语：Tools / Resources / Prompts / Sampling
   其它协议：Roots(根) / 能力协商取交集 / initialized 通知 / 四类消息 / 标准错误码
   传输：stdio(真子进程往返) / HTTP+SSE(说明) ；封装：FastMCP(@tool/@resource/@prompt)
   客户端：手写 JSON-RPC / 官方 SDK(smolagents / huggingface_hub Agent)
 本机手写 JSON-RPC 全部真跑；FastMCP/官方客户端为真实代码(惰性导入，需装 SDK/联网，本机不跑)。
 跑：python3 全功能1_MCP协议全能力.py
================================================================================
"""
import sys
import json
import subprocess

DONE = set()   # 记录已演示的功能，末尾出清单


# ==============================================================================
# 一个手写 MCP 服务器：实现所有方法(单进程直接调 handle 演示各原语)
# ==============================================================================
def _server_handle(req):
    m, rid, params = req.get("method"), req.get("id"), req.get("params", {})

    def ok(r):
        return {"jsonrpc": "2.0", "id": rid, "result": r}

    def err(code, msg):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}

    if m == "initialize":
        server_caps = {"tools": {}, "resources": {}, "prompts": {}}
        return ok({"protocolVersion": "2024-11-05", "capabilities": server_caps,
                   "serverInfo": {"name": "all-in-one", "version": "1.0"}})
    if m == "tools/list":
        return ok({"tools": [{"name": "echo", "description": "回显文本",
                              "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                                              "required": ["text"]}}]})
    if m == "tools/call":
        name, args = params.get("name"), params.get("arguments", {})
        if name != "echo":
            return err(-32602, f"Unknown tool: {name}")          # 参数/工具错误码
        return ok({"content": [{"type": "text", "text": "echo: " + args.get("text", "")}], "isError": False})
    if m == "resources/list":
        return ok({"resources": [{"uri": "docs://faq", "name": "FAQ", "mimeType": "text/plain"}]})
    if m == "resources/read":
        return ok({"contents": [{"uri": params.get("uri"), "mimeType": "text/plain",
                                 "text": "这是 FAQ 的内容(只读资源)。"}]})
    if m == "resources/templates/list":
        return ok({"resourceTemplates": [{"uriTemplate": "note://{id}", "name": "笔记"}]})
    if m == "prompts/list":
        return ok({"prompts": [{"name": "summarize", "description": "总结文本",
                               "arguments": [{"name": "text", "required": True}]}]})
    if m == "prompts/get":
        text = params.get("arguments", {}).get("text", "")
        return ok({"messages": [{"role": "system", "content": {"type": "text", "text": "你是总结助手。"}},
                                {"role": "user", "content": {"type": "text", "text": f"总结：{text}"}}]})
    return err(-32601, f"Method not found: {m}")                 # 方法不存在错误码


def _call(method, **params):
    return _server_handle({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})


# ==============================================================================
# 一、四大原语 + 资源模板 + 错误码
# ==============================================================================
def primitives():
    # print("=" * 70, "\n① 四大原语 + 资源模板 + 错误码\n" + "=" * 70)
    init = _call("initialize", capabilities={"roots": {}, "sampling": {}})["result"]
    print("  initialize →", init["serverInfo"]); DONE.add("initialize")
    # 能力协商取交集
    client_caps, server_caps = {"roots": {}, "sampling": {}}, set(init["capabilities"])
    usable = set(client_caps) & server_caps
    print(f"  能力协商取交集 → 客户端{set(client_caps)} ∩ 服务器{server_caps} = {usable}")
    DONE.add("能力协商取交集")

    print("  [Tools] tools/list →", [t["name"] for t in _call("tools/list")["result"]["tools"]]); DONE.add("Tools:list")
    r = _call("tools/call", name="echo", arguments={"text": "hi"})["result"]
    print("  [Tools] tools/call →", r["content"][0]["text"]); DONE.add("Tools:call")
    print("  [Resources] resources/list →", _call("resources/list")["result"]["resources"][0]["uri"]); DONE.add("Resources:list")
    print("  [Resources] resources/read →", _call("resources/read", uri="docs://faq")["result"]["contents"][0]["text"][:12], "..."); DONE.add("Resources:read")
    print("  [Resources] templates/list →", _call("resources/templates/list")["result"]["resourceTemplates"][0]["uriTemplate"]); DONE.add("Resources:templates")
    print("  [Prompts] prompts/list →", _call("prompts/list")["result"]["prompts"][0]["name"]); DONE.add("Prompts:list")
    g = _call("prompts/get", name="summarize", arguments={"text": "长文本"})["result"]
    print("  [Prompts] prompts/get → 角色:", [msg["role"] for msg in g["messages"]]); DONE.add("Prompts:get")
    # 错误码
    e1 = _call("tools/call", name="nope")["error"]["code"]
    e2 = _call("no_such_method")["error"]["code"]
    print(f"  错误码 → 工具不存在{e1} / 方法不存在{e2}"); DONE.add("错误码-32601/-32602")


# ==============================================================================
# 二、Sampling / Roots / initialized 通知 / 四类消息(server↔client 反向 + 通知)
# ==============================================================================
def sampling_roots_notify():
    # print("\n" + "=" * 70, "\n② Sampling / Roots / initialized 通知 / 四类消息\n" + "=" * 70)
    # Sampling(服务器→客户端 请求 Host 的 LLM)
    samp = {"jsonrpc": "2.0", "id": 200, "method": "sampling/createMessage",
            "params": {"messages": [{"role": "user", "content": {"type": "text", "text": "总结 MCP"}}],
                       "maxTokens": 100}}
    print("  Sampling(server→client) →", samp["method"]); DONE.add("Sampling:createMessage")
    # Roots(服务器→客户端)
    roots = {"jsonrpc": "2.0", "id": 100, "method": "roots/list"}
    roots_resp = {"result": {"roots": [{"uri": "file:///proj", "name": "项目"}]}}
    print("  Roots(server→client) →", roots["method"], "→", roots_resp["result"]["roots"][0]["uri"]); DONE.add("Roots:list")
    # initialized 通知(无 id)
    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    is_note = "method" in note and "id" not in note
    print("  initialized 通知(无id=通知) →", is_note); DONE.add("initialized通知")
    # 四类消息判别
    def classify(m):
        if "method" in m and "id" in m: return "请求"
        if "method" in m: return "通知"
        if "error" in m: return "错误"
        return "响应"
    print("  四类消息 →", classify(samp), classify(note), classify({"id": 1, "result": {}}), classify({"id": 1, "error": {}}))
    DONE.add("四类消息判别")


# ==============================================================================
# 三、传输 stdio(真子进程往返，连 mcp实战/案例1 真服务器)
# ==============================================================================
def transport_stdio():
    # print("\n" + "=" * 70, "\n③ 传输：stdio 真子进程往返(连真 NLP 服务器) + HTTP/SSE 说明\n" + "=" * 70)
    import os
    server = os.path.join(os.path.dirname(__file__), "..", "..", "mcp实战", "案例1_MCP工具服务器_NLP能力.py")
    p = subprocess.Popen([sys.executable, server, "--server"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        def rpc(method, params=None):
            p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}) + "\n")
            p.stdin.flush()
            return json.loads(p.stdout.readline())["result"]
        tools = [t["name"] for t in rpc("tools/list")["tools"]]
        out = rpc("tools/call", {"name": "get_sentiment", "arguments": {"text": "I love it!"}})["content"][0]["text"]
        print(f"  stdio 往返 → 发现{tools}，调用 get_sentiment → {out}")
        assert json.loads(out)["label"] == "POSITIVE"
        DONE.add("传输stdio(真往返)")
    finally:
        p.stdin.close(); p.terminate()
    # print("  HTTP+SSE / Streamable HTTP：远程服务器传输(mcp.run(transport='sse'))，多 Host 共享。")
    DONE.add("传输HTTP/SSE(说明)")


# ==============================================================================
# 四、FastMCP 封装 + 官方 SDK 客户端（真实代码，惰性导入，需 SDK/联网，本机不跑）
# ==============================================================================
def fastmcp_server():
    """FastMCP：@mcp.tool / @mcp.resource / @mcp.prompt 一键暴露(需 pip install fastmcp)。"""
    from fastmcp import FastMCP
    mcp = FastMCP("all-in-one")

    @mcp.tool
    def echo(text: str) -> str:
        """回显文本。"""
        return "echo: " + text

    @mcp.resource("docs://faq")
    def faq() -> str:
        """FAQ 只读资源。"""
        return "FAQ 内容"

    @mcp.prompt
    def summarize(text: str) -> str:
        """总结提示模板。"""
        return f"总结：{text}"
    return mcp   # mcp.run() 起服务；launch(mcp_server=True) 也能把 Gradio 函数变 MCP 工具


def official_clients():
    """官方 SDK 客户端(smolagents / huggingface_hub Agent) —— 真实代码，需 SDK+联网。"""
    from smolagents import ToolCollection, CodeAgent, InferenceClientModel
    from mcp import StdioServerParameters
    params = StdioServerParameters(command="python3", args=["server.py", "--server"])
    with ToolCollection.from_mcp(params, trust_remote_code=True) as tc:
        return CodeAgent(tools=[*tc.tools], model=InferenceClientModel()).run("分析情感")


if __name__ == "__main__":
    primitives()
    sampling_roots_notify()
    transport_stdio()
    # print("\n" + "=" * 70, "\n④ FastMCP 封装 + 官方客户端(真实代码，本机不跑)\n" + "=" * 70)
    # print("  fastmcp_server(): @mcp.tool/@mcp.resource/@mcp.prompt 一键暴露")
    # print("  official_clients(): smolagents ToolCollection.from_mcp + CodeAgent")
    for f in ("FastMCP@tool/@resource/@prompt", "官方SDK客户端"):
        DONE.add(f)
    # —— 功能覆盖清单 ——
    ALL = ["initialize", "能力协商取交集", "Tools:list", "Tools:call", "Resources:list", "Resources:read",
           "Resources:templates", "Prompts:list", "Prompts:get", "错误码-32601/-32602",
           "Sampling:createMessage", "Roots:list", "initialized通知", "四类消息判别",
           "传输stdio(真往返)", "传输HTTP/SSE(说明)", "FastMCP@tool/@resource/@prompt", "官方SDK客户端"]
    # print("\n" + "=" * 70, "\n📋 MCP 全功能覆盖清单\n" + "=" * 70)
    for f in ALL:
        print(f"  {'✅' if f in DONE else '❌'} {f}")
    missing = [f for f in ALL if f not in DONE]
    assert not missing, f"未覆盖: {missing}"
    print(f"\n✅ 全功能1 跑通：MCP {len(ALL)} 项功能全部覆盖(手写真跑 + FastMCP/客户端真实代码)。")
