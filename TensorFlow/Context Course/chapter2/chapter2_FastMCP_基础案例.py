"""
 Context Course · Ch2 · 基础案例：最小 MCP 服务器(纯 python 实现协议,机制真实,可跑)
 MCP 的本质 = 用 JSON-RPC 暴露工具:Host 发 tools/list 发现工具、发 tools/call 调用。
 这里不装任何库,纯 python 把这套"发现→调用"协议跑通;真实生产用 FastMCP 一行装饰器代劳(见文末)。
 跑：python3 本文件
"""
import inspect
import json

_TOOLS = {}


def tool(fn):
    """迷你 @mcp.tool():从类型注解 + docstring 自动生成工具 schema,登记到工具表。"""
    sig = inspect.signature(fn)
    _TOOLS[fn.__name__] = {
        "fn": fn,
        "schema": {
            "name": fn.__name__,
            "description": (fn.__doc__ or "").strip(),
            "inputSchema": {
                "type": "object",
                "properties": {p: {"type": "integer" if par.annotation is int else "string"}
                               for p, par in sig.parameters.items()},
                "required": list(sig.parameters),
            },
        },
    }
    return fn


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


def handle(request):
    """MCP 服务器主逻辑:按 JSON-RPC method 路由(真实 FastMCP 内部也是这套)。"""
    method, params = request["method"], request.get("params", {})
    if method == "tools/list":                       # ① Host 发现有哪些工具及其 schema
        return {"tools": [t["schema"] for t in _TOOLS.values()]}
    if method == "tools/call":                       # ② Host 按 schema 组参数请求调用
        result = _TOOLS[params["name"]]["fn"](**params["arguments"])
        return {"content": [{"type": "text", "text": str(result)}]}
    return {"error": f"unknown method {method}"}


if __name__ == "__main__":
    listed = handle({"method": "tools/list"})
    print("① 发现工具:", json.dumps([t["name"] for t in listed["tools"]], ensure_ascii=False))
    called = handle({"method": "tools/call", "params": {"name": "add", "arguments": {"a": 2, "b": 3}}})
    print("② 调用 add(2,3) →", called["content"][0]["text"])
    assert called["content"][0]["text"] == "5"
    print('✅ 纯 python 跑通 MCP 的 tools/list + tools/call;真实生产:@mcp.tool()+mcp.run() 起 stdio 服务')
