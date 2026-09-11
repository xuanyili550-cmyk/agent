"""
================================================================================
 生产综合案例4 · LLM 推理服务 + Agent 网关（真实生产：本地 LLM + 工具调用 + Web/API）
================================================================================
 一个带工具的对话服务：用户聊天，Agent 判断要不要调工具(MCP 的 NLP 能力)，否则本地 LLM 直接答。
 覆盖章节功能：
   [Ch1 推理部署]  mlx-lm 本地 LLM(Qwen2.5-0.5B)，流式输出；OpenAI 兼容 API(换 vLLM 只改后端)
   [MCP]           情感/NER 工具通过 MCP 协议连 ../../mcp实战/案例1 的服务器
   [Agent]         Agent 决定：调 MCP 工具分析 还是 LLM 直接对话(function calling 思路)
   [Ch9 Gradio]    ChatInterface 流式聊天；FastAPI 出 OpenAI 兼容 /v1/chat/completions
 生产要点：模型/服务单例、流式、OpenAI 兼容(前端不锁厂商)、工具即插即用(MCP)。
 运行：
   python3 生产案例4_LLM推理服务与Agent网关.py smoke   # LLM 生成 + MCP 工具 + Agent 路由 自检
   python3 生产案例4_LLM推理服务与Agent网关.py         # 起流式聊天 Web
   python3 生产案例4_LLM推理服务与Agent网关.py api     # 起 OpenAI 兼容 API(端口 8000)
================================================================================
"""
import os
import re
import sys
import json
import subprocess
import gradio as gr

_M = {}
MCP_SERVER = os.path.join(os.path.dirname(__file__), "..", "..", "mcp实战",
                          "案例1_MCP工具服务器_NLP能力.py")


# ---- [Ch1] mlx-lm 本地 LLM ----
def _load():
    if "m" not in _M:
        from mlx_lm import load
        _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    return _M


def llm_complete(messages, n=200):
    from mlx_lm import generate
    s = _load()
    return generate(s["m"], s["t"], prompt=s["t"].apply_chat_template(messages, add_generation_prompt=True),
                    max_tokens=n, verbose=False).strip()


def llm_stream(messages, n=200):
    from mlx_lm import stream_generate
    s = _load()
    for chunk in stream_generate(s["m"], s["t"],
                                 prompt=s["t"].apply_chat_template(messages, add_generation_prompt=True),
                                 max_tokens=n):
        yield chunk.text


# ---- [MCP] NLP 工具 ----
def _mcp():
    if "mcp" not in _M:
        _M["mcp"] = subprocess.Popen([sys.executable, MCP_SERVER, "--server"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        _M["mid"] = 0
    return _M["mcp"]


def mcp_tool(name, **args):
    p = _mcp(); _M["mid"] += 1
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": _M["mid"], "method": "tools/call",
                              "params": {"name": name, "arguments": args}}, ensure_ascii=False) + "\n")
    p.stdin.flush()
    return json.loads(p.stdout.readline())["result"]["content"][0]["text"]


# ---- [Agent] 路由：分析请求→MCP 工具；否则 LLM 对话 ----
def agent_reply(message, history):
    if re.search(r"情感|sentiment", message, re.I):
        target = re.sub(r".*[:：]", "", message).strip() or message
        return f"🔧 我调用了情感分析工具(MCP)：\n{mcp_tool('get_sentiment', text=target)}"
    if re.search(r"实体|entit|人名|公司", message, re.I):
        target = re.sub(r".*[:：]", "", message).strip() or message
        return f"🔧 我调用了实体识别工具(MCP)：\n{mcp_tool('extract_entities', text=target)}"
    return None                                              # 交给 LLM 直接答


def chat_fn(message, history):
    tool_out = agent_reply(message, history)
    if tool_out is not None:
        yield tool_out
        return
    messages = (history or []) + [{"role": "user", "content": message}]   # Gradio6: history=messages字典列表
    acc = ""
    for piece in llm_stream(messages, n=200):
        acc += piece
        yield acc


def build_demo():
    return gr.ChatInterface(
        fn=chat_fn, title="LLM 推理服务 + Agent 网关",
        description="本地 mlx-lm 对话(流式)；说“分析情感：xxx / 抽实体：xxx”会自动调 MCP 工具。",
        examples=["用一句话介绍机器学习", "分析情感：This product is fantastic!", "抽实体：Tim Cook at Apple"],
        analytics_enabled=False)


def make_api():
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    app = FastAPI(title="本地 LLM 服务(OpenAI 兼容)")

    class Req(BaseModel):
        model: str = "local"
        messages: list
        max_tokens: int = 200
        stream: bool = False

    @app.post("/v1/chat/completions")
    def cc(r: Req):
        if r.stream:
            def sse():
                for p in llm_stream(r.messages, r.max_tokens):
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': p}}]}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")
        return {"choices": [{"message": {"role": "assistant", "content": llm_complete(r.messages, r.max_tokens)}}]}
    return app


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if arg == "smoke":
        try:
            print("LLM:", llm_complete([{"role": "user", "content": "用一句话介绍机器学习"}], 40)[:50])
            print("流式:", "".join(list(llm_stream([{"role": "user", "content": "数到3"}], 15)))[:40])
            print("Agent→MCP情感:", agent_reply("分析情感：This is fantastic!", [])[:60])
            print("Agent→MCP实体:", agent_reply("抽实体：Tim Cook at Apple", [])[:60])
            assert "POSITIVE" in agent_reply("分析情感：I love it!", [])
            assert agent_reply("你好啊", []) is None            # 普通对话交给 LLM
            build_demo()
            print("✅ 生产案例4 自检通过：mlx-lm 流式 + MCP 工具 + Agent 路由 + ChatInterface。")
        finally:
            if "mcp" in _M:
                _M["mcp"].stdin.close(); _M["mcp"].terminate()
    elif arg == "api":
        import uvicorn
        uvicorn.run(make_api(), host="0.0.0.0", port=8000)
    else:
        build_demo().launch()
