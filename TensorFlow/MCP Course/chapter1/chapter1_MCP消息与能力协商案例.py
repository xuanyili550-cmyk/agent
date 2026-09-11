"""
================================================================================
 MCP Course · Chapter 1 · 案例：亲手构造 MCP 消息 + 能力协商（可运行·纯 Python 无依赖）
================================================================================
 chapter1 的学习笔记只是把“真实 MCP 消息长什么样”打印出来给你看；本案例更进一步——
 我们亲手写一个「消息构造器 + 校验器」，把 MCP 通信里最容易被忽略的两件事讲清楚：

   1) JSON-RPC 2.0 的消息骨架到底有哪些字段、请求/响应/通知/错误各长什么样、
      为什么每条请求都要带 id（用来把响应和请求配对，异步时不会串线）。
   2) initialize 握手里的「能力协商 capabilities」到底在协商什么——
      客户端和服务器各自声明“我支持哪些原语”，最终只有【双方都支持】的能力才能用。
      这是 MCP 的关键设计：不假设对方支持什么，一切按协商结果来。

 全程不连网、不起子进程、不装任何 SDK：纯粹把协议消息当数据结构来玩，
 用 assert 证明我们构造的每条消息都符合 JSON-RPC 2.0 / MCP 规范。

 运行：python3 chapter1_MCP消息与能力协商案例.py
================================================================================
"""

import json

JSONRPC = "2.0"          # JSON-RPC 版本号，MCP 固定用 2.0
PROTOCOL = "2024-11-05"  # MCP 协议版本，握手时双方要对齐


# ==============================================================================
# 一、消息构造器：把“怎么拼一条合法 JSON-RPC 消息”封装成函数（这就是所有 SDK 底层在做的事）
# ==============================================================================
def make_request(req_id, method, params=None):
    """请求：必须带 id（用于和响应配对）+ method（要调用的方法）。"""
    return {"jsonrpc": JSONRPC, "id": req_id, "method": method, "params": params or {}}


def make_notification(method, params=None):
    """通知：和请求几乎一样，但【没有 id】——因为它不需要回应（发完就算，如 initialized）。"""
    return {"jsonrpc": JSONRPC, "method": method, "params": params or {}}


def make_response(req_id, result):
    """成功响应：id 要和对应请求一致，携带 result。"""
    return {"jsonrpc": JSONRPC, "id": req_id, "result": result}


def make_error(req_id, code, message):
    """错误响应：用 error 取代 result；code 是标准错误码（-32601=方法不存在 等）。"""
    return {"jsonrpc": JSONRPC, "id": req_id, "error": {"code": code, "message": message}}


# ==============================================================================
# 二、消息校验器：证明我们拼出来的东西真的合法（避免“自以为对”）
# ==============================================================================
def validate_message(msg):
    """按 JSON-RPC 2.0 规范校验一条消息，返回它的类型标签。不合法就抛异常。"""
    assert msg.get("jsonrpc") == "2.0", "jsonrpc 字段必须是 '2.0'"
    has_id = "id" in msg
    if "method" in msg:
        # 有 method → 是请求(带 id) 或 通知(无 id)
        return "request" if has_id else "notification"
    if "result" in msg:
        assert has_id, "响应必须带 id（用来配对是哪条请求的回应）"
        assert "error" not in msg, "一条消息不能同时有 result 和 error"
        return "response"
    if "error" in msg:
        assert has_id, "错误响应也要带 id"
        e = msg["error"]
        assert "code" in e and "message" in e, "error 必须含 code + message"
        return "error"
    raise AssertionError("既不是请求/通知，也不是响应/错误 —— 非法消息")


# ==============================================================================
# 三、能力协商：MCP 握手的核心——只有双方都支持的能力才启用
# ==============================================================================
def negotiate_capabilities(client_caps, server_caps):
    """把客户端和服务器各自声明的能力取交集：这就是本次会话真正可用的能力集合。
    真实场景：服务器可能提供 tools+resources+prompts，但客户端只支持 tools，
    那这次会话就只能用 tools —— 不能对客户端用它不懂的原语。"""
    return sorted(set(client_caps) & set(server_caps))


def show(title, obj):
    print(f"\n── {title} ──")
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def main():
    print("=" * 72)
    print(" 案例：亲手构造 MCP 消息 + 做一次能力协商（全程纯数据结构，可 assert 验证）")
    print("=" * 72)

    # ---- ① 握手 initialize：客户端声明自己支持的能力 ----
    init_req = make_request(1, "initialize", {
        "protocolVersion": PROTOCOL,
        "capabilities": {"tools": {}, "prompts": {}},   # 这个客户端支持 工具 + 提示
        "clientInfo": {"name": "my-host", "version": "1.0"},
    })
    show("① 客户端 → 服务器：initialize 请求", init_req)
    assert validate_message(init_req) == "request"

    # 服务器回应：声明自己支持的能力（注意：它多支持 resources，但少了 prompts）
    init_resp = make_response(1, {
        "protocolVersion": PROTOCOL,
        "capabilities": {"tools": {}, "resources": {}},  # 服务器支持 工具 + 资源
        "serverInfo": {"name": "demo-server", "version": "1.0.0"},
    })
    show("① 服务器 → 客户端：initialize 响应", init_resp)
    assert validate_message(init_resp) == "response"

    # ---- ② 能力协商：取交集，得出本次会话真正能用的能力 ----
    client_caps = list(init_req["params"]["capabilities"])
    server_caps = list(init_resp["result"]["capabilities"])
    usable = negotiate_capabilities(client_caps, server_caps)
    print("\n── ② 能力协商结果 ──")
    print(f"   客户端支持: {client_caps}")
    print(f"   服务器支持: {server_caps}")
    print(f"   ★双方都支持(本次会话可用): {usable}")
    print("   说明: prompts 只有客户端支持、resources 只有服务器支持 → 都用不了；只有 tools 能用。")
    assert usable == ["tools"], "交集应只剩 tools"

    # ---- ③ initialized 通知：握手完成后客户端发一条【无 id】通知，告诉服务器可以开工了 ----
    inited = make_notification("notifications/initialized")
    show("③ 客户端 → 服务器：initialized 通知(无 id，不需要回应)", inited)
    assert validate_message(inited) == "notification"
    assert "id" not in inited, "通知不能带 id"

    # ---- ④ 用协商出来的 tools 能力：发现工具 + 调用工具 ----
    list_req = make_request(2, "tools/list")
    list_resp = make_response(2, {"tools": [{
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "inputSchema": {"type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]},
    }]})
    show("④ 发现工具 tools/list 响应", list_resp)
    assert validate_message(list_req) == "request"
    assert validate_message(list_resp) == "response"

    call_req = make_request(3, "tools/call",
                            {"name": "get_weather", "arguments": {"location": "Beijing"}})
    call_resp = make_response(3, {
        "content": [{"type": "text", "text": json.dumps({"temperature": 22, "conditions": "Cloudy"})}],
        "isError": False,
    })
    show("④ 调用工具 tools/call 请求", call_req)
    show("④ 调用工具 tools/call 响应", call_resp)
    assert validate_message(call_resp) == "response"

    # ---- ⑤ 错误响应：调用一个不存在的方法，服务器要用标准错误码回应 ----
    err_resp = make_error(4, -32601, "Method not found: tools/summon_dragon")
    show("⑤ 未知方法 → 标准错误响应(code=-32601)", err_resp)
    assert validate_message(err_resp) == "error"

    # ---- 全量自检：把上面所有消息再统一校验一遍，并核对请求/响应 id 配对 ----
    pairs = [(init_req, init_resp), (list_req, list_resp), (call_req, call_resp)]
    for req, resp in pairs:
        assert req["id"] == resp["id"], "响应 id 必须和请求 id 一致，否则无法配对"

    print("\n" + "=" * 72)
    print(" ✅ 自检通过：")
    print("   · 请求/通知/响应/错误 四类消息全部符合 JSON-RPC 2.0 规范；")
    print("   · 请求与响应的 id 一一配对成功；")
    print("   · 能力协商正确地把可用能力收敛为双方交集 ['tools']。")
    print(" 这就是任何 MCP 库(FastMCP 等)在底层默默替你做的事 —— 你已经亲手做了一遍。")
    print("=" * 72)


if __name__ == "__main__":
    main()
