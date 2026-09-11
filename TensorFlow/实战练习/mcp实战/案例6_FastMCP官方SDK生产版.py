"""
================================================================================
 案例6 · 官方 FastMCP SDK 生产版（把 NLP 能力暴露成 MCP 工具）
================================================================================
 前面案例1-5 用“纯手写 JSON-RPC”是为了懂协议底层；本机现在装了官方 SDK(fastmcp 3.4.7)，
 生产上就该用它——几行搞定，自动生成 schema、支持 stdio/HTTP 传输、和 Claude Desktop 直连。
 手写版 vs FastMCP：说的是同一套 JSON-RPC 消息，FastMCP 帮你把样板代码全省了。

 整合：把 情感分析(Ch1) + NER(Ch6/7) 用 @mcp.tool 暴露成工具。函数签名+docstring 自动变成
       工具的 name/description/inputSchema，客户端/Claude 一 tools/list 就能发现并调用。

 用法：
   python3 案例6_FastMCP官方SDK生产版.py smoke   # 构建服务器+直接调工具函数自检(不起服务)
   python3 案例6_FastMCP官方SDK生产版.py          # 真起 MCP 服务器(stdio)，供 Claude Desktop/客户端连
================================================================================
"""
import sys
from fastmcp import FastMCP

mcp = FastMCP("nlp-tools-server")     # 一个 MCP 服务器

# 模型惰性单例(生产：加载一次全程复用，别每次调用都重载)
_cache = {}
def _sent():
    if "s" not in _cache:
        from transformers import pipeline
        _cache["s"] = pipeline("sentiment-analysis",
                               model="distilbert-base-uncased-finetuned-sst-2-english")
    return _cache["s"]
def _ner():
    if "n" not in _cache:
        from transformers import pipeline
        _cache["n"] = pipeline("token-classification",
                               model="huggingface-course/bert-finetuned-ner",
                               aggregation_strategy="simple")
    return _cache["n"]


@mcp.tool                              # ← 这一个装饰器就把函数变成了 MCP 工具
def get_sentiment(text: str) -> dict:
    """Analyze the sentiment of a text. Returns label (POSITIVE/NEGATIVE) and score."""
    r = _sent()(text[:1000])[0]
    return {"label": r["label"], "score": round(float(r["score"]), 4)}


@mcp.tool
def extract_entities(text: str) -> dict:
    """Extract named entities (people, organizations, locations) from a text."""
    ents = _ner()(text)
    return {"entities": [{"group": e["entity_group"], "word": e["word"],
                          "score": round(float(e["score"]), 4)} for e in ents]}


# 也能一行暴露资源(只读数据)和提示(模板)：
@mcp.resource("config://server-info")
def server_info() -> str:
    """Basic info about this NLP MCP server."""
    return '{"name": "nlp-tools-server", "tools": ["get_sentiment", "extract_entities"]}'


def smoke():
    # FastMCP 工具本质就是普通函数，直接调用验证业务逻辑
    s = get_sentiment("This update is fantastic, I love it!")
    e = extract_entities("Sylvain works at Hugging Face in New York.")
    # print("  get_sentiment →", s)
    # print("  extract_entities →", e)
    assert s["label"] == "POSITIVE"
    assert any(x["group"] == "PER" for x in e["entities"])
    print("  ✅ FastMCP 服务器构建成功，工具函数自检通过。")
    # print("  真起服务：python3", sys.argv[0].split('/')[-1], "  (默认 stdio；mcp.run(transport='sse') 走 HTTP)")


# ==============================================================================
# 生产/集成
# ==============================================================================
# · 传输：默认 stdio(本地，Claude Desktop 直连)；mcp.run(transport="sse") 走 HTTP/SSE(远程共享)。
# · 接入 Claude Desktop：编辑 claude_desktop_config.json：
#     {"mcpServers": {"nlp": {"command": "python3", "args": ["/绝对路径/案例6_FastMCP官方SDK生产版.py"]}}}
#   重启 Claude Desktop → 它 tools/list 发现 get_sentiment/extract_entities → 对话里能自动调用。
# · 生产：模型服务和 MCP 服务器可分离(MCP 只做工具编排，重推理转发给 vLLM，见 ../chapter实战/生产架构)。
#   鉴权/审计/限流在 MCP 服务器或前置网关做；工具能执行真实操作，务必最小权限 + 参数校验。
#
# 面试：Q FastMCP 和手写 JSON-RPC 关系？A 同一套 MCP 协议消息，FastMCP 是官方封装(装饰器自动生成
#      schema/处理传输)，生产用它；手写是为了懂底层。
#      Q MCP 和 function calling 区别？A function calling 是“模型输出要调哪个函数”；MCP 是“工具/数据
#      怎么被标准化地暴露和发现”的协议(一次实现处处可插)，两者配合：MCP 提供工具 → 模型 function calling 调它。

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke()
    else:
        mcp.run()      # 真起 MCP 服务器(stdio)，阻塞等待客户端/Claude 连接
