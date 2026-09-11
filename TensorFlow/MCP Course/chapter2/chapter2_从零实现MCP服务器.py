"""
================================================================================
 MCP Course · Chapter 2 · 从零实现一个真正的 MCP 服务器 + 客户端（stdio/JSON-RPC）
================================================================================
 不装任何 SDK，用纯 Python 实现 MCP 协议的核心：stdio 传输 + JSON-RPC 2.0 消息，
 覆盖真实握手与调用流程：initialize → tools/list → tools/call。跑通就说明你真懂 MCP 底层。

 MCP 架构：Host(如 Claude Desktop) 里的 Client 通过传输层连到 Server，Server 暴露
   工具(Tools)/资源(Resources)/提示(Prompts)。这里实现 stdio 传输(服务器= 一个子进程，
   client 通过它的 stdin/stdout 收发 JSON-RPC，每行一条 JSON)。

 用法：
   python3 chapter2_从零实现MCP服务器.py            # 跑客户端：它会把自己当服务器 spawn 起来，走完整流程
   python3 chapter2_从零实现MCP服务器.py --server    # 只作为 MCP 服务器(读 stdin 的 JSON-RPC，回 stdout)

 生产对照：真实项目用官方 SDK `pip install mcp`(FastMCP)几行就能起服务器(见 chapter4)。
 这里手写是为了看清协议本身；两者说的是同一套 JSON-RPC 消息。
================================================================================
"""

import json
import subprocess
import sys


# ==============================================================================
# 一、服务器侧：暴露工具 + 处理 JSON-RPC 请求
# ==============================================================================
# 工具注册表：每个工具有 名字/描述/输入 JSON Schema(供客户端发现) + 实际实现函数
def _get_weather(location: str):
    return {"temperature": 72, "conditions": "Sunny", "humidity": 45, "location": location}

def _add(a: float, b: float):
    return {"sum": a + b}

TOOLS = {
    "get_weather": {
        "description": "Get the current weather for a location.",
        "inputSchema": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
        "func": _get_weather,
    },
    "add_numbers": {
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        "func": _add,
    },
}


def handle_request(req: dict) -> dict:
    """MCP 服务器核心：按 method 分发。返回 JSON-RPC 响应。"""
    method, req_id = req.get("method"), req.get("id")
    params = req.get("params", {})

    def ok(result):
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def err(code, msg):
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}

    if method == "initialize":
        # 握手：双方交换协议版本与能力(capabilities)。这里声明我们提供 tools。
        return ok({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "demo-weather-server", "version": "1.0.0"},
        })

    if method == "tools/list":
        # ★动态发现：客户端问“你有哪些工具”，返回名字/描述/输入 schema
        return ok({"tools": [
            {"name": n, "description": t["description"], "inputSchema": t["inputSchema"]}
            for n, t in TOOLS.items()
        ]})

    if method == "tools/call":
        # ★调用工具：params={"name":..., "arguments":{...}}，返回 content 列表
        name = params.get("name")
        args = params.get("arguments", {})
        if name not in TOOLS:
            return err(-32602, f"Unknown tool: {name}")
        try:
            result = TOOLS[name]["func"](**args)
            return ok({"content": [{"type": "text", "text": json.dumps(result)}],
                       "isError": False})
        except Exception as e:
            return ok({"content": [{"type": "text", "text": str(e)}], "isError": True})

    return err(-32601, f"Method not found: {method}")


def run_server():
    """作为 MCP 服务器运行：逐行读 stdin 的 JSON-RPC 请求，把响应写回 stdout。"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_request(req)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()          # 必须 flush，否则客户端读不到(stdio 传输的关键)


# ==============================================================================
# 二、客户端侧：把服务器当子进程 spawn，走完整 MCP 流程
# ==============================================================================
class MCPClient:
    def __init__(self, server_cmd):
        # stdio 传输：启动服务器子进程，通过它的 stdin/stdout 通信
        self.proc = subprocess.Popen(
            server_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline())
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        return resp["result"]

    def close(self):
        self.proc.stdin.close()
        self.proc.terminate()


def run_client():
    print("=" * 72 + "\n MCP 客户端：连接服务器，走 initialize → 发现工具 → 调用工具\n" + "=" * 72)
    client = MCPClient([sys.executable, __file__, "--server"])
    try:
        # ① 握手
        info = client.call("initialize", {"protocolVersion": "2024-11-05",
                                          "capabilities": {}, "clientInfo": {"name": "demo-client"}})
        print(f"\n① initialize 握手成功 → 服务器 = {info['serverInfo']['name']} "
              f"v{info['serverInfo']['version']}，能力 = {list(info['capabilities'])}")

        # ② 动态发现工具(tools/list)
        tools = client.call("tools/list")["tools"]
        print(f"\n② tools/list 发现 {len(tools)} 个工具：")
        for t in tools:
            print(f"   · {t['name']}: {t['description']}")
            print(f"     输入参数: {list(t['inputSchema']['properties'])}")

        # ③ 调用工具(tools/call)
        print("\n③ tools/call 调用工具：")
        r1 = client.call("tools/call", {"name": "get_weather", "arguments": {"location": "Paris"}})
        print("   get_weather(Paris) →", r1["content"][0]["text"])
        r2 = client.call("tools/call", {"name": "add_numbers", "arguments": {"a": 2, "b": 3}})
        print("   add_numbers(2,3)   →", r2["content"][0]["text"])

        # ④ 错误处理：调用不存在的工具
        try:
            client.call("tools/call", {"name": "no_such_tool", "arguments": {}})
        except RuntimeError as e:
            print("\n④ 调用不存在的工具 → 服务器正确报错:", e)

        # 自检
        assert json.loads(r1["content"][0]["text"])["conditions"] == "Sunny"
        assert json.loads(r2["content"][0]["text"])["sum"] == 5
        print("\n✅ 自检通过：真实 MCP 流程(stdio + JSON-RPC)跑通——握手/发现/调用/报错全对。")
        print("   这就是 Claude Desktop 等 Host 连接 MCP 服务器时底层发生的事。")
    finally:
        client.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_server()
    else:
        run_client()
