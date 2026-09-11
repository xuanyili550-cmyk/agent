"""
================================================================================
 MCP Course · Chapter 4 · 综合实战：MCP 驱动的 Agent 循环 + 官方 SDK + 集成部署
================================================================================
 把前几章串起来，展示 MCP 在真实产品里怎么用：
   一、MCP Agent 循环：用户提问 → 发现工具 → (LLM)选工具 → 调用 → 用结果回答。
       这正是 Claude Desktop / IDE 里“AI 会用工具”的底层机制。
   二、官方 SDK(FastMCP) 生产写法：几行起一个 MCP 服务器(参考代码，需 pip install mcp)。
   三、集成：把你的 MCP 服务器注册进 Claude Desktop(claude_desktop_config.json)。
   四、生产注意事项：安全、传输选型、能力协商。

 用法：python3 chapter4_综合实战_Agent与官方SDK.py
   会连接 chapter2 的真实 MCP 服务器，跑一个“根据问题自动选工具并调用”的 Agent 循环。
================================================================================
"""

import json
import os
import re
import subprocess
import sys

# 复用 chapter2 那个真实 MCP 服务器(演示“连接到一个已有的 MCP 服务器”)
CH2_SERVER = os.path.join(os.path.dirname(__file__), "..", "chapter2",
                          "chapter2_从零实现MCP服务器.py")


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


# ==============================================================================
# 一、MCP Agent 循环（用户问题 → 选工具 → 调用 → 回答）
# ==============================================================================
def choose_tool(query, tools):
    """模拟 LLM 的‘工具选择’：真实 Agent 里是把 query + 工具列表(schema) 发给 LLM，
    让它输出要调哪个工具、参数是什么。这里用关键词规则替代 LLM，逻辑结构一样。"""
    q = query.lower()
    if "weather" in q or "天气" in q:
        loc = "Paris" if "paris" in q else "Beijing"
        return "get_weather", {"location": loc}
    if "+" in q or "plus" in q or "加" in q:
        nums = [int(s) for s in re.findall(r"\d+", q)]   # 用正则抠出所有数字(不受标点影响)
        if len(nums) >= 2:
            return "add_numbers", {"a": nums[0], "b": nums[1]}
    return None, None


def agent_loop():
    print("=" * 72 + "\n 一、MCP Agent 循环：AI 根据问题自动发现+选择+调用工具\n" + "=" * 72)
    client = MCPClient([sys.executable, CH2_SERVER, "--server"])
    try:
        client.call("initialize", {})
        tools = client.call("tools/list")["tools"]          # ① 发现工具
        print(f"Agent 已连上 MCP 服务器，可用工具: {[t['name'] for t in tools]}\n")

        for query in ["What's the weather in Paris?", "What is 12 plus 30?", "Tell me a joke"]:
            print(f"用户: {query}")
            name, args = choose_tool(query, tools)          # ② (LLM)选工具
            if name is None:
                print("  Agent: (没有合适的工具，直接用 LLM 回答)\n")
                continue
            result = client.call("tools/call", {"name": name, "arguments": args})  # ③ 调用
            data = json.loads(result["content"][0]["text"])
            # ④ 用工具结果组织回答(真实里把结果回填给 LLM 生成自然语言)
            if name == "get_weather":
                print(f"  Agent → 调用 {name}({args}) → 回答: {args['location']} "
                      f"当前 {data['temperature']}°F, {data['conditions']}\n")
            else:
                print(f"  Agent → 调用 {name}({args}) → 回答: 结果是 {data['sum']}\n")
        print("✅ Agent 循环跑通：这就是 MCP 让 AI‘会用工具’的完整闭环(发现→选择→调用→回答)。")
    finally:
        client.close()


# ==============================================================================
# 二、官方 SDK(FastMCP) 生产写法（参考代码，需 pip install mcp）
# ==============================================================================
def show_fastmcp_reference():
    print("\n" + "=" * 72 + "\n 二、官方 SDK：FastMCP 起一个生产 MCP 服务器(几行)\n" + "=" * 72)
    print('''  # pip install mcp
  from mcp.server.fastmcp import FastMCP
  mcp = FastMCP("weather-server")

  @mcp.tool()                                  # 工具：函数签名+docstring 自动生成 schema
  def get_weather(location: str) -> dict:
      """Get current weather for a location."""
      return {"temperature": 72, "conditions": "Sunny"}

  @mcp.resource("config://settings")           # 资源：只读数据
  def settings() -> str:
      return '{"theme": "dark"}'

  @mcp.prompt()                                # 提示：模板
  def code_review(code: str) -> str:
      return f"Review this code:\\n{code}"

  if __name__ == "__main__":
      mcp.run()                                # 默认 stdio；mcp.run(transport="sse") 走 HTTP
  → FastMCP 底层收发的，就是 chapter2/3 手写的那套 JSON-RPC(initialize/tools/list/tools/call)。
  → Gradio 也能一行起 MCP：demo.launch(mcp_server=True)(需 pip install "gradio[mcp]")。''')


# ==============================================================================
# 三、集成进 Claude Desktop
# ==============================================================================
def show_claude_desktop_config():
    print("\n" + "=" * 72 + "\n 三、把你的 MCP 服务器注册进 Claude Desktop\n" + "=" * 72)
    config = {
        "mcpServers": {
            "weather": {
                "command": "python3",
                "args": ["/absolute/path/to/chapter2_从零实现MCP服务器.py", "--server"],
            }
        }
    }
    print("  编辑 claude_desktop_config.json，加入：\n")
    print("  " + json.dumps(config, indent=2, ensure_ascii=False).replace("\n", "\n  "))
    print("\n  重启 Claude Desktop → 它会 spawn 你的服务器、tools/list 发现工具，")
    print("  之后你在对话里问‘北京天气’，Claude 就会自动调用你的 get_weather 工具。")


# ==============================================================================
# 四、生产注意事项
# ==============================================================================
def show_production_notes():
    print("\n" + "=" * 72 + "\n 四、生产注意事项\n" + "=" * 72)
    print("""  · 安全：工具能执行真实操作(删文件/调 API/花钱)，务必校验参数、最小权限、危险操作要确认。
  · 传输选型：本地个人工具用 stdio；团队/远程共享用 HTTP+SSE(带鉴权)。
  · 能力协商：initialize 时双方交换 capabilities，别假设对方支持某原语，按协商结果用。
  · 错误处理：tools/call 出错要返回 isError=True + 说明，别让服务器崩(客户端才能优雅处理)。
  · 版本：protocolVersion 要对齐(如 2024-11-05)，不兼容时降级或报错。""")


if __name__ == "__main__":
    agent_loop()
    show_fastmcp_reference()
    show_claude_desktop_config()
    show_production_notes()
