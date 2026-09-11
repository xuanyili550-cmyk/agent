"""
================================================================================
 MCP Course 挖空练习 · 实现一个 MCP 服务器的 JSON-RPC 分发
================================================================================
 玩法：
   1) 把每个 ______ 换成正确代码（凭记忆，别翻笔记）
   2) 运行：python3 挖空练习_实现MCP服务器.py   （纯 Python，无依赖，秒出）
   3) 没填的报 NameError；卡住 → 文件底部「答案区」
 目标：实现 MCP 服务器的核心——按 JSON-RPC 的 method 分发 initialize / tools/list /
       tools/call，并返回符合 MCP 规范的响应。(完整版见 chapter2)
================================================================================
"""
import json

# 一个工具：名字 → (描述, 实现)
def _add(a, b):
    return {"sum": a + b}

TOOLS = {"add_numbers": {"description": "Add two numbers.", "func": _add,
                         "inputSchema": {"type": "object",
                                         "properties": {"a": {"type": "number"},
                                                        "b": {"type": "number"}},
                                         "required": ["a", "b"]}}}


def handle_request(req):
    # 练习1：从请求里取 method、id、params（JSON-RPC 请求的三个字段）
    method = req.get("method")
    rid = req.get("id")
    params = req.get("params", {})

    def ok(result):
        # 练习2：JSON-RPC 成功响应的固定结构（jsonrpc/id/result 三个键）
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    if method == "initialize":
        # 练习3：握手要声明协议版本 + capabilities（这里我们有 tools）
        return ok({"protocolVersion": "2024-11-05",
                   "capabilities": {"tools": {}},
                   "serverInfo": {"name": "practice", "version": "1.0.0"}})

    if method == "tools/list":
        # 练习4：返回工具清单，每个含 name/description/inputSchema
        return ok({"tools": [{"name": n, "description": t["description"],
                              "inputSchema": t["inputSchema"]} for n, t in TOOLS.items()]})

    if method == "tools/call":
        # 练习5：取工具名和参数，调用对应函数，把结果包成 content 列表
        name = params["name"]
        args = params.get("arguments", {})
        result = TOOLS[name]["func"](**args)
        return ok({"content": [{"type": "text", "text": json.dumps(result)}],
                   "isError": False})

    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}}


# ---- 自测：模拟客户端依次发 initialize / tools/list / tools/call ----
init = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
lst = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
call = handle_request({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                       "params": {"name": "add_numbers", "arguments": {"a": 2, "b": 3}}})

print("initialize →", init["result"]["serverInfo"])
print("tools/list →", [t["name"] for t in lst["result"]["tools"]])
print("tools/call →", call["result"]["content"][0]["text"])

assert init["result"]["capabilities"]["tools"] == {}
assert lst["result"]["tools"][0]["name"] == "add_numbers"
assert json.loads(call["result"]["content"][0]["text"])["sum"] == 5
print("\n自检通过 ✅：你的 MCP 服务器正确处理了 握手/发现/调用 三步 JSON-RPC。")


# ==============================================================================
#  答案区（卡住再看！）
# ------------------------------------------------------------------------------
#  1: method = req.get("method"); rid = req.get("id"); params = req.get("params", {})
#  2: return {"jsonrpc": "2.0", "id": rid, "result": result}
#  3: {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {...}}
#  4: {"tools": [{"name": n, "description": ..., "inputSchema": ...} for ...]}
#  5: name = params["name"]; result = TOOLS[name]["func"](**args);
#     ok({"content": [{"type": "text", "text": json.dumps(result)}], "isError": False})
# ==============================================================================
