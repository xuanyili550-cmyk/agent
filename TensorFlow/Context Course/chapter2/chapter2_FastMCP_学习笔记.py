"""
================================================================================
 Context Course · Chapter 2 · 用 FastMCP 写 MCP 服务器（学习笔记 · 工具契约 python 可跑）
================================================================================
 一句话：MCP 让工具"被描述→被发现→被调用";FastMCP 用装饰器几行把普通函数暴露成 MCP 工具。
 本章讲：
   ① 工具三要素 = name + description + 输入 schema —— LLM 靠这三样决定"调不调、传什么"(本文件纯 python 演示这套契约)。
   ② FastMCP：@mcp.tool() 自动从类型注解/文档字符串生成 schema,mcp.run() 起 stdio/HTTP 服务。
   ③ 和手写 JSON-RPC 是同一套协议(见 实战练习/mcp实战)。
 要点：description 写得好坏直接决定 LLM 选工具准不准;schema 由类型注解自动生成。
 跑：python3 chapter2_FastMCP_学习笔记.py   （契约演示纯 python 真跑;真实 server 见 serve_real 🟡需 mcp 库)
================================================================================
"""
import inspect


def mcp_tool(fn):
    """迷你版 @tool：从函数签名 + docstring 自动生成"工具契约"(模仿 FastMCP 干的事)。"""
    sig = inspect.signature(fn)
    fn.contract = {
        "name": fn.__name__,
        "description": (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "",
        "inputSchema": {p: (par.annotation.__name__ if par.annotation is not inspect._empty else "any")
                        for p, par in sig.parameters.items()},
    }
    return fn


@mcp_tool
def add(a: int, b: int) -> int:
    """把两个整数相加。"""
    return a + b


def serve_real():   # 🟡 需 pip install "mcp[cli]",默认不调用
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("calculator")
    mcp.tool()(add)          # 真实 FastMCP：装饰后 mcp.run() 起服务
    return mcp


def main():
    c = add.contract
    assert add(2, 3) == 5
    assert c["name"] == "add" and c["inputSchema"] == {"a": "int", "b": "int"}
    print(f"✅ Ch2 跑通：工具契约 {c}")
    print("   真实生产：@mcp.tool() 自动生成同样的 schema,mcp.run() 起 stdio/HTTP 服务给 Host 连。")
    # 面试：Q LLM 靠什么选工具? A name+description+inputSchema; Q FastMCP 省了什么? A 手写 JSON-RPC 样板。


if __name__ == "__main__":
    main()
