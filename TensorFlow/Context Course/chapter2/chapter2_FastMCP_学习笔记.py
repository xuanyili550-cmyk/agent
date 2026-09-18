"""
================================================================================
 Context Course · Chapter 2 · 用 FastMCP 写 MCP 服务器（学习笔记 · 协议全流程 python 可跑）
================================================================================
 一句话：MCP 让工具"被描述→被发现→被调用";底层是 JSON-RPC,FastMCP 用装饰器把普通函数几行暴露成工具。
 本章用纯 python 把整套协议跑通(不装库、机制真实)：
   ① 工具三要素 = name + description + inputSchema —— LLM 靠这三样决定"调不调、传什么"。
   ② 服务器主循环:按 JSON-RPC method 路由 initialize / tools/list / tools/call。
   ③ FastMCP 干的事:@mcp.tool() 自动从类型注解/docstring 生成 ①,mcp.run() 起 ② 的 stdio/HTTP 服务。
 要点：description 写得好坏直接决定 LLM 选工具准不准;schema 由类型注解自动生成,你不用手写 JSON-RPC 样板。
 跑：python3 chapter2_FastMCP_学习笔记.py   （协议全流程纯 python 真跑;真实 server 见 serve_real 🟡需 mcp 库)
================================================================================
"""
import inspect

_TOOLS = {}


def mcp_tool(fn):
    """迷你版 @tool:从函数签名 + docstring 自动生成"工具契约"并登记(模仿 FastMCP)。"""
    sig = inspect.signature(fn)
    _TOOLS[fn.__name__] = {
        "fn": fn,
        "schema": {
            "name": fn.__name__,
            "description": (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "",
            "inputSchema": {p: (par.annotation.__name__ if par.annotation is not inspect._empty else "any")
                            for p, par in sig.parameters.items()},
        },
    }
    return fn


@mcp_tool
def add(a: int, b: int) -> int:
    """把两个整数相加。"""
    return a + b


def server(request):
    """② 服务器主循环:按 JSON-RPC method 路由(真实 FastMCP 内部同构)。"""
    method = request["method"]
    if method == "initialize":                       # 握手:交换协议版本/服务器信息
        return {"protocolVersion": "2024-11-05", "serverInfo": {"name": "calculator"}}
    if method == "tools/list":                       # 发现:回全部工具 schema
        return {"tools": [t["schema"] for t in _TOOLS.values()]}
    if method == "tools/call":                       # 调用:按名找函数、拆参数执行
        p = request["params"]
        return {"content": [{"type": "text", "text": str(_TOOLS[p["name"]]["fn"](**p["arguments"]))}]}
    return {"error": f"unknown method {method}"}


def serve_real():   # 🟡 需 pip install "mcp[cli]",默认不调用
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("calculator")
    mcp.tool()(add)          # 真实 FastMCP:装饰后 mcp.run() 起 stdio/HTTP 服务
    return mcp


def main():
    assert add(2, 3) == 5
    hello = server({"method": "initialize"})                              # ① 握手
    tools = server({"method": "tools/list"})["tools"]                     # ② 发现
    assert hello["serverInfo"]["name"] == "calculator"
    assert tools[0]["name"] == "add" and tools[0]["inputSchema"] == {"a": "int", "b": "int"}
    out = server({"method": "tools/call",                                 # ③ 调用
                  "params": {"name": "add", "arguments": {"a": 2, "b": 3}}})
    assert out["content"][0]["text"] == "5"
    print(f"✅ Ch2 跑通:initialize → tools/list={[t['name'] for t in tools]} → tools/call add(2,3)={out['content'][0]['text']}")
    print("   真实生产:@mcp.tool() 自动生成同样 schema,mcp.run() 起 stdio/HTTP 服务给 Host 连。")
    # 面试:Q LLM 靠什么选工具? A name+description+inputSchema; Q MCP 底层协议? A JSON-RPC 的 tools/list+tools/call; Q FastMCP 省了什么? A 手写样板。


if __name__ == "__main__":
    main()
