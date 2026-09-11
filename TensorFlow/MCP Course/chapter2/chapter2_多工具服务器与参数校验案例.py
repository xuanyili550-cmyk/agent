"""
================================================================================
 MCP Course · Chapter 2 · 案例：多工具服务器 + 真·参数校验（可运行·纯 Python 手写 JSON-RPC）
================================================================================
 chapter2 的笔记实现了一个最小的“两个工具”服务器。本案例把它升级成更接近生产的样子，
 重点补上真实 MCP 服务器绕不开的两件事：

   1) 多工具：一次注册好几个工具（计算器 / 字符串处理 / 回显），演示 tools/list 动态发现
      如何让客户端(以及 LLM)自己“看菜单点菜”。
   2) ★参数校验：LLM 生成的 arguments 经常缺字段、类型错、超范围。生产服务器【必须】在真正
      执行前，拿工具的 inputSchema 校验参数，非法就返回 isError=True 的结构化错误，
      而不是让函数崩掉。本案例手写了一个迷你 JSON-Schema 校验器（type / required / enum / 范围），
      这正是官方 SDK 用 pydantic 自动帮你做的那层保护。

 传输仍是真实的 stdio + JSON-RPC 2.0：客户端把本文件当子进程 spawn 起来，逐行收发 JSON。

 运行：
   python3 chapter2_多工具服务器与参数校验案例.py           # 跑客户端(自动 spawn 服务器)，走全流程
   python3 chapter2_多工具服务器与参数校验案例.py --server   # 仅作为 MCP 服务器(读 stdin / 写 stdout)
================================================================================
"""

import json
import subprocess
import sys


# ==============================================================================
# 一、工具实现（每个工具 = inputSchema 供发现/校验 + func 真正执行）
# ==============================================================================
def _calculate(a, b, op):
    ops = {"add": a + b, "sub": a - b, "mul": a * b}
    if op == "div":
        if b == 0:
            raise ValueError("除数不能为 0")           # 业务错误：会被包成 isError=True 返回
        return {"result": a / b}
    return {"result": ops[op]}


def _string_op(text, action):
    if action == "upper":
        return {"result": text.upper()}
    if action == "reverse":
        return {"result": text[::-1]}
    return {"result": len(text)}                       # action == "length"


def _echo(message):
    return {"echo": message}


TOOLS = {
    "calculate": {
        "description": "对两个数做四则运算(add/sub/mul/div)。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                # enum：限定 op 只能是这四个值之一 —— 校验器会拦截其它值
                "op": {"type": "string", "enum": ["add", "sub", "mul", "div"]},
            },
            "required": ["a", "b", "op"],
        },
        "func": _calculate,
    },
    "string_op": {
        "description": "字符串处理：转大写/反转/求长度。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},   # 演示范围类约束
                "action": {"type": "string", "enum": ["upper", "reverse", "length"]},
            },
            "required": ["text", "action"],
        },
        "func": _string_op,
    },
    "echo": {
        "description": "原样回显一段消息(最简单的工具，用来做连通性测试)。",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        "func": _echo,
    },
}


# ==============================================================================
# 二、★迷你 JSON-Schema 校验器：在执行前守门（生产 MCP 服务器的关键保护层）
# ==============================================================================
_TYPE_MAP = {"string": str, "number": (int, float), "integer": int,
             "boolean": bool, "object": dict, "array": list}


def validate_args(schema, args):
    """按 inputSchema 校验 args，返回错误信息列表(空列表=合法)。
    覆盖最常用的约束：required / type / enum / minLength / 数值范围。
    真实 SDK 用完整 JSON-Schema，但拦截的思路完全一样：先验证，再执行。"""
    errors = []
    props = schema.get("properties", {})

    # required：必填字段不能缺
    for req in schema.get("required", []):
        if req not in args:
            errors.append(f"缺少必填参数 '{req}'")

    for key, val in args.items():
        if key not in props:
            errors.append(f"未知参数 '{key}'")
            continue
        spec = props[key]
        # type：类型要对（注意 bool 是 int 的子类，需单独排除，避免把 True 当数字放过）
        expected = spec.get("type")
        if expected in _TYPE_MAP:
            pytype = _TYPE_MAP[expected]
            if expected in ("number", "integer") and isinstance(val, bool):
                errors.append(f"参数 '{key}' 类型应为 {expected}，但收到 bool")
            elif not isinstance(val, pytype):
                errors.append(f"参数 '{key}' 类型应为 {expected}，但收到 {type(val).__name__}")
                continue
        # enum：取值必须在白名单里
        if "enum" in spec and val not in spec["enum"]:
            errors.append(f"参数 '{key}' 取值应为 {spec['enum']} 之一，但收到 '{val}'")
        # minLength / 数值范围
        if spec.get("minLength") is not None and isinstance(val, str) and len(val) < spec["minLength"]:
            errors.append(f"参数 '{key}' 长度应 >= {spec['minLength']}")
        if spec.get("minimum") is not None and isinstance(val, (int, float)) and val < spec["minimum"]:
            errors.append(f"参数 '{key}' 应 >= {spec['minimum']}")
    return errors


# ==============================================================================
# 三、服务器：按 JSON-RPC method 分发（校验 → 执行 → 结构化返回）
# ==============================================================================
def handle_request(req):
    method, rid, params = req.get("method"), req.get("id"), req.get("params", {})
    ok = lambda r: {"jsonrpc": "2.0", "id": rid, "result": r}
    err = lambda c, m: {"jsonrpc": "2.0", "id": rid, "error": {"code": c, "message": m}}

    if method == "initialize":
        return ok({"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                   "serverInfo": {"name": "multi-tool-server", "version": "1.0.0"}})

    if method == "tools/list":
        return ok({"tools": [{"name": n, "description": t["description"],
                              "inputSchema": t["inputSchema"]} for n, t in TOOLS.items()]})

    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments", {})
        if name not in TOOLS:
            return err(-32602, f"未知工具: {name}")            # 协议级错误：工具根本不存在
        # ★先校验参数，非法就返回 isError=True(工具级错误)，绝不让 func 直接崩
        problems = validate_args(TOOLS[name]["inputSchema"], args)
        if problems:
            return ok({"content": [{"type": "text", "text": "参数校验失败: " + "; ".join(problems)}],
                       "isError": True})
        try:
            result = TOOLS[name]["func"](**args)
            return ok({"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                       "isError": False})
        except Exception as e:                                   # 业务异常也包成 isError，不让服务器挂
            return ok({"content": [{"type": "text", "text": f"执行出错: {e}"}], "isError": True})

    return err(-32601, f"方法不存在: {method}")


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
        sys.stdout.flush()   # stdio 传输必须 flush，否则客户端读不到


# ==============================================================================
# 四、客户端：spawn 服务器子进程，走完整流程 + 覆盖“非法参数”的边界
# ==============================================================================
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
    print("=" * 72)
    print(" 多工具 MCP 服务器 + 参数校验：发现工具 → 正常调用 → 触发各类参数错误")
    print("=" * 72)
    c = MCPClient([sys.executable, __file__, "--server"])
    try:
        c.call("initialize", {})
        tools = c.call("tools/list")["tools"]
        print(f"\n① tools/list 发现 {len(tools)} 个工具：")
        for t in tools:
            print(f"   · {t['name']}: {t['description']}")

        print("\n② 正常调用（参数合法）：")
        r1 = c.call("tools/call", {"name": "calculate", "arguments": {"a": 12, "b": 30, "op": "add"}})
        print("   calculate(12+30) →", r1["content"][0]["text"])
        r2 = c.call("tools/call", {"name": "string_op", "arguments": {"text": "MCP", "action": "reverse"}})
        print("   string_op('MCP', reverse) →", r2["content"][0]["text"])
        r3 = c.call("tools/call", {"name": "echo", "arguments": {"message": "hi"}})
        print("   echo('hi') →", r3["content"][0]["text"])
        assert json.loads(r1["content"][0]["text"])["result"] == 42
        assert json.loads(r2["content"][0]["text"])["result"] == "PCM"

        print("\n③ 参数校验拦截（这些都不会让服务器崩，而是返回 isError=True）：")
        # (a) 缺必填参数 op
        e1 = c.call("tools/call", {"name": "calculate", "arguments": {"a": 1, "b": 2}})
        print("   缺 op    →", e1["content"][0]["text"], "| isError =", e1["isError"])
        assert e1["isError"] and "op" in e1["content"][0]["text"]
        # (b) 类型错误：a 传了字符串
        e2 = c.call("tools/call", {"name": "calculate", "arguments": {"a": "x", "b": 2, "op": "add"}})
        print("   a 非数字 →", e2["content"][0]["text"], "| isError =", e2["isError"])
        assert e2["isError"]
        # (c) enum 越界：op 不在白名单
        e3 = c.call("tools/call", {"name": "calculate", "arguments": {"a": 1, "b": 2, "op": "pow"}})
        print("   op 非法  →", e3["content"][0]["text"], "| isError =", e3["isError"])
        assert e3["isError"] and "pow" in e3["content"][0]["text"]
        # (d) 业务异常：除以 0（参数合法但运行时出错，同样被安全捕获）
        e4 = c.call("tools/call", {"name": "calculate", "arguments": {"a": 1, "b": 0, "op": "div"}})
        print("   除以 0   →", e4["content"][0]["text"], "| isError =", e4["isError"])
        assert e4["isError"] and "除数" in e4["content"][0]["text"]

        print("\n④ 协议级错误（工具不存在 → JSON-RPC error，不是 isError）：")
        try:
            c.call("tools/call", {"name": "no_such", "arguments": {}})
        except RuntimeError as ex:
            print("   调用 no_such →", ex)

        print("\n" + "=" * 72)
        print(" ✅ 自检通过：多工具发现正常；4 类参数/运行错误全部被校验层安全拦截并结构化返回；")
        print("    工具不存在走 JSON-RPC 协议错误。这就是生产 MCP 服务器该有的健壮性。")
        print("=" * 72)
    finally:
        c.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_server()
    else:
        run_client()
