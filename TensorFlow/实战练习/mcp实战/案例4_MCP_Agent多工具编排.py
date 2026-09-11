"""
================================================================================
 MCP 实战 · 案例4 · MCP Agent：连接多个 MCP 服务器，自动发现→选工具→调用→回答
================================================================================
 这是 MCP 的最终形态：一个 Agent(充当 Host)同时连接【多个】MCP 服务器，把它们的工具
 汇总成一张"能力表"，再根据用户问题自动挑工具、调用、综合结果作答。

 本例复用前面两个真实服务器(不重复造轮子，体现 MCP"工具实现一次、到处可连"):
   · 案例1_MCP工具服务器_NLP能力.py   → 提供 get_sentiment / extract_entities
   · 案例2_MCP服务器_语义检索.py       → 提供 search_knowledge
 Agent 用 subprocess 各起一个子进程(stdio 传输)，对每个都 initialize + tools/list，
 建立【工具名 → 属于哪个服务器】的路由表；调用时把 tools/call 转发到对应服务器。

 选工具策略：本例用【规则式】(关键词/意图) 便于离线可复现并跑得快。
 !! 真实生产里这一步由 LLM 做 function calling !!：把 tools/list 的 schema 塞进 LLM，
    让它读用户问题后自己决定"调哪个工具、填什么参数"，Host 再通过 MCP 执行。规则式只是替身。

 用法：python3 案例4_MCP_Agent多工具编排.py
================================================================================
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 要连接的 MCP 服务器(复用案例1/案例2 的文件，各以 --server 模式起子进程)
SERVER_FILES = {
    "nlp":       os.path.join(HERE, "案例1_MCP工具服务器_NLP能力.py"),
    "retrieval": os.path.join(HERE, "案例2_MCP服务器_语义检索.py"),
}


# ==============================================================================
# 一、MCP 客户端连接(每个服务器一条 stdio 连接)
# ==============================================================================
class MCPConnection:
    def __init__(self, name, server_file):
        self.name = name
        self.proc = subprocess.Popen(
            [sys.executable, server_file, "--server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._id = 0
        self.server_info = self.call("initialize", {"protocolVersion": "2024-11-05",
                                                    "capabilities": {}, "clientInfo": {"name": "mcp-agent"}})

    def call(self, method, params=None):
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline())
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        return resp["result"]

    def close(self):
        self.proc.stdin.close()
        self.proc.terminate()


# ==============================================================================
# 二、Agent：聚合多服务器工具 + 路由 + 规则式选工具 + 综合回答
# ==============================================================================
class MCPAgent:
    def __init__(self):
        self.conns = {}
        self.tool_route = {}   # tool_name -> conn(哪台服务器提供)
        self.tools = {}        # tool_name -> schema(供"选工具"和 LLM function calling 用)
        for name, f in SERVER_FILES.items():
            conn = MCPConnection(name, f)
            self.conns[name] = conn
            for t in conn.call("tools/list")["tools"]:
                self.tool_route[t["name"]] = conn
                self.tools[t["name"]] = t

    def call_tool(self, tool_name, arguments):
        """把 tools/call 路由到提供该工具的服务器。"""
        conn = self.tool_route[tool_name]
        r = conn.call("tools/call", {"name": tool_name, "arguments": arguments})
        if r.get("isError"):
            raise RuntimeError(r["content"][0]["text"])
        return json.loads(r["content"][0]["text"])

    def plan(self, query):
        """规则式选工具(真实里由 LLM function calling 决定)。返回 [(tool, args), ...]。"""
        q = query.lower()
        steps = []
        # 有情绪词/感叹 → 先判情感(客服里用于分流)
        if any(w in q for w in ["love", "hate", "great", "terrible", "frustrat", "awful",
                                "!", "worst", "amazing", "angry", "happy"]):
            steps.append(("get_sentiment", {"text": query}))
        # 出现大写专有名词(人/机构/地点)→ 抽实体
        if any(tok[:1].isupper() for tok in query.split()[1:]):   # 跳过首词大写(句首)
            steps.append(("extract_entities", {"text": query}))
        # 只要是个问题/求助 → 检索知识库找答案(RAG)
        if any(w in q for w in ["how", "what", "why", "where", "can i", "?", "help", "reset",
                                "refund", "crash", "storage", "password", "export"]):
            steps.append(("search_knowledge", {"query": query, "top_k": 1}))
        if not steps:   # 兜底：至少检索一下
            steps.append(("search_knowledge", {"query": query, "top_k": 1}))
        return steps

    def answer(self, query):
        """执行计划、调用工具、把结果综合成一段回答(真实里这步也交给 LLM 生成)。"""
        results = []
        for tool, args in self.plan(query):
            out = self.call_tool(tool, args)
            results.append((tool, out))
        # 综合(规则式模板；生产里把 results 拼进提示让 LLM 写自然语言回复)
        parts = []
        for tool, out in results:
            if tool == "get_sentiment":
                parts.append(f"情感={out['label']}({out['score']})")
            elif tool == "extract_entities":
                ents = ", ".join(f"{e['group']}:{e['word']}" for e in out["entities"]) or "无"
                parts.append(f"实体=[{ents}]")
            elif tool == "search_knowledge":
                top = out["results"][0]
                parts.append(f"知识库答案(相似度{top['score']}): {top['text']}")
        return results, " | ".join(parts)

    def close(self):
        for c in self.conns.values():
            c.close()


# ==============================================================================
# 三、演示
# ==============================================================================
def main():
    # print("=" * 72 + "\n MCP 案例4：Agent 连接多个 MCP 服务器，自动发现→选工具→调用→回答\n" + "=" * 72)
    agent = MCPAgent()
    try:
        print(f"\n① 已连接 {len(agent.conns)} 个 MCP 服务器：")
        for name, conn in agent.conns.items():
            print(f"   · [{name}] {conn.server_info['serverInfo']['name']}")
        print(f"\n② 聚合发现 {len(agent.tools)} 个工具(跨服务器统一能力表)：")
        for tn, conn in agent.tool_route.items():
            print(f"   · {tn}  ← 来自服务器[{conn.name}]")

        queries = [
            "I was charged twice and I'm really frustrated, how do I get a refund from PayPal?",
            "How much storage does a free account get?",
            "The app from Apple keeps crashing when I upload photos, this is terrible!",
        ]
        # print("\n③ Agent 自动编排(规则式选工具；真实里由 LLM function calling)：")
        checks = []
        for q in queries:
            results, summary = agent.answer(q)
            used = [t for t, _ in results]
            # print("\n" + "─" * 68)
            print(f"用户: {q}")
            print(f"  Agent 选用工具: {used}")
            print(f"  综合回答: {summary}")
            checks.append(results)

        # 自检：第一个问题应触发情感+实体+检索三个工具，且检索命中 refund
        used0 = [t for t, _ in checks[0]]
        assert "get_sentiment" in used0 and "search_knowledge" in used0
        kb0 = [o for t, o in checks[0] if t == "search_knowledge"][0]
        assert "refund" in kb0["results"][0]["text"].lower()
        # print("\n" + "─" * 68)
        print("✅ 自检通过：Agent 跨 2 个 MCP 服务器聚合工具，按问题自动选并调用，综合作答正确。")
    finally:
        agent.close()


# ==============================================================================
# 生产要点(上云怎么搭)
# ==============================================================================
# · 选工具交给 LLM：把 agent.tools 的 name/description/inputSchema 转成 LLM 的 tools 参数(function
#   calling / MCP 原生支持)，模型自己决定调哪个、填什么参数、要不要多轮，比规则鲁棒得多。
# · 多服务器管理：服务器可能崩/超时，要对每条连接加健康检查、超时、重连、熔断；工具命名冲突要加服务器前缀。
# · 编排循环：真实 Agent 是"思考→调用→观察→再思考"的多轮循环(ReAct)，直到能回答;要限制最大步数防死循环。
# · 权限与审计：不同用户能看到/调用的工具集不同(RBAC)；每次 tools/call 记录用户/工具/参数/结果做审计。
# · 上下文预算：多工具结果会堆很多 token，需裁剪/摘要后再喂 LLM，避免超窗和成本失控。

# ==============================================================================
# 面试题(MCP Agent 编排)
# ==============================================================================
# Q: 一个 Host 连多个 MCP 服务器，工具重名怎么办？
# A: 用"服务器命名空间"限定，如 nlp.get_sentiment / retrieval.search_knowledge；路由表按限定名找连接，
#    避免不同服务器同名工具冲突。
# Q: Agent 怎么决定调用哪个工具？规则 vs LLM？
# A: 生产用 LLM function calling：把 tools/list 的 schema 给模型，它读用户意图后输出要调的工具名+参数
#    (可多个/多轮)。规则式(本例)只在意图窄、要可复现/省成本时用。关键：工具的 description 写得好坏直接
#    决定 LLM 选得准不准。
# Q: MCP 相比传统 Agent 框架(如把工具硬编码进 LangChain)的核心优势?
# A: 标准化 + 解耦。工具作为独立 MCP 服务器进程存在，跨语言/跨团队/跨 Host 复用；换 Agent 框架或换主
#    模型都不用改工具；工具热插拔、独立部署与扩缩容。
# Q: 多工具编排最容易出的生产事故?
# A: ①无超时/无最大步数导致死循环烧钱 ②工具结果不裁剪撑爆上下文 ③某服务器挂了没熔断拖垮整链
#    ④工具返回不可信数据被 LLM 当指令执行(提示注入)。

if __name__ == "__main__":
    main()
