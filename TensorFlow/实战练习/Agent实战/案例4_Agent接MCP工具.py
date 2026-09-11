"""
================================================================================
 Agent 实战 · 案例4 · Agent 通过 MCP 连远程工具（整合 mcp实战）
================================================================================
 整合 MCP：Agent 不必把工具都写死在自己进程里——通过 MCP 协议【连接工具服务器、动态发现工具、
 按需调用】。工具部署在哪、怎么加载，Agent 完全不用管(能力即插即用)。
 这里让 Agent 连 ../mcp实战/案例1 那个“NLP 工具服务器”(情感 Ch1 + NER Ch6/7)。

 本机现实：
   · smoke：用【手写 MCP stdio 客户端】spawn 那个 MCP 服务器 → initialize → tools/list(发现) →
     tools/call(真调用)。这就是“Agent 侧发现并调用 MCP 工具”的完整流程(本地真跑)。
   · 生产：smolagents 有官方封装 ToolCollection.from_mcp，一行把 MCP 工具变成 Agent 工具——
     但需 `pip install 'smolagents[mcp]'`(本机未装 mcpadapt)，故作真实代码参考。
 跑：python3 案例4_Agent接MCP工具.py smoke
================================================================================
"""
import os
import sys
import json
import subprocess

MCP_SERVER = os.path.join(os.path.dirname(__file__), "..", "mcp实战",
                          "案例1_MCP工具服务器_NLP能力.py")


# ==============================================================================
# 手写 MCP stdio 客户端（Agent 侧：发现 + 调用远程工具）
# ==============================================================================
class MCPStdioClient:
    def __init__(self, server_path):
        self.proc = subprocess.Popen([sys.executable, server_path, "--server"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline())
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        return resp["result"]

    def list_tools(self):
        return self.call("tools/list")["tools"]

    def call_tool(self, name, **args):
        r = self.call("tools/call", {"name": name, "arguments": args})
        return r["content"][0]["text"]

    def close(self):
        self.proc.stdin.close(); self.proc.terminate()


# ==============================================================================
# 生产：smolagents 官方封装（真实代码，惰性导入）——需 pip install 'smolagents[mcp]'(装 mcpadapt)，本机未装
# ==============================================================================
def run_via_smolagents_mcp(question="分析情感并抽实体：Tim Cook works at Apple in Cupertino."):
    """一行把 MCP 服务器的工具变成 Agent 工具，让 LLM 自己编排(生产写法)。
    需 pip install 'smolagents[mcp]' + HF Token；本机未装 mcpadapt，故不跑。"""
    from smolagents import ToolCollection, CodeAgent, InferenceClientModel   # 惰性导入
    from mcp import StdioServerParameters
    params = StdioServerParameters(command="python3", args=[MCP_SERVER, "--server"])
    # from_mcp 自动 tools/list 发现、按 inputSchema 生成工具签名；也支持 SSE：{"url":".../mcp/sse","transport":"sse"}
    with ToolCollection.from_mcp(params, trust_remote_code=True) as tc:
        agent = CodeAgent(tools=[*tc.tools], model=InferenceClientModel())
        return agent.run(question)


def smoke():
    # 手写 MCP 客户端：连接 mcp实战/案例1 的 NLP 工具服务器(initialize → tools/list 发现 → tools/call 调用)
    client = MCPStdioClient(MCP_SERVER)
    try:
        info = client.call("initialize", {"protocolVersion": "2024-11-05",
                                          "capabilities": {}, "clientInfo": {"name": "agent-host"}})
        # ① initialize -> serverInfo.name 为 MCP 服务器名
        tools = client.list_tools()
        # ② tools/list 动态发现工具：应含 get_sentiment / extract_entities
        # ③ tools/call 真实调用(第一次会惰性加载模型，稍等)，期望：
        #   get_sentiment("This new update is fantastic...") -> label POSITIVE
        #   extract_entities("Tim Cook works at Apple in Cupertino.") -> 含 PER 实体
        s = client.call_tool("get_sentiment", text="This new update is fantastic, I love it!")
        n = client.call_tool("extract_entities", text="Tim Cook works at Apple in Cupertino.")
        assert json.loads(s)["label"] == "POSITIVE"
        assert any(e["group"] == "PER" for e in json.loads(n)["entities"])
        print("\n✅ 案例4 跑通：Agent 侧通过 MCP 发现并调用了远程 NLP 工具(情感 Ch1 + NER Ch6/7)。")
        # 生产用 smolagents ToolCollection.from_mcp 一行接入(见 run_via_smolagents_mcp)
        # print("面试：Q Agent 为什么要用 MCP(不把工具写死)? A 工具即插即用/热插拔、跨进程/跨团队复用、")
        # print("     升级工具不用改 Agent 代码；MCP 提供工具 → LLM function calling 决定调用。")
    finally:
        client.close()


if __name__ == "__main__":
    smoke()
    # print("\n—— 生产：真实函数 run_via_smolagents_mcp() 用 smolagents ToolCollection.from_mcp"
          # " 一行接入 MCP 工具(需 smolagents[mcp]+Token，本机未装 mcpadapt 不跑) ——")
