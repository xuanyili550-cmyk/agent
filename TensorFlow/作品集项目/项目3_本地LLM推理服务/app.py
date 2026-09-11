"""
作品集项目3 · 本地 LLM 推理服务（OpenAI 兼容 API + 聊天 Web，Mac 上真跑）
--------------------------------------------------------------------------------
 一句话：在 Mac(mps)上用 mlx-lm 跑一个本地 LLM，暴露成【OpenAI 兼容的 /v1/chat/completions API】
        + 一个流式聊天 Web。概念对齐 vLLM(生产)，本机用 mlx-lm 替代(Mac 上 vLLM 装不了)。
 生产要点：模型单例、流式输出(打字机)、OpenAI 兼容(客户端/前端无缝切换、不锁厂商)、健康探针。
 换生产：把 mlx-lm 换成 vLLM(见 ../../实战练习/chapter实战/生产01/05)，API 契约完全一样，前端不用改。
 运行：
   python3 app.py smoke   # 加载模型 + 生成一次 + 流式一次 自检(不起服务)
   python3 app.py         # 起聊天 Web(Gradio ChatInterface，流式)
   python3 app.py api     # 起 OpenAI 兼容 FastAPI 服务(/v1/chat/completions，端口 8000)
"""
import sys
import json
import time
import gradio as gr

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
_M = {}


def _load():
    if "model" not in _M:
        from mlx_lm import load
        _M["model"], _M["tok"] = load(MODEL_ID)             # 单例：只加载一次
    return _M


def _to_prompt(messages):
    s = _load()
    return s["tok"].apply_chat_template(messages, add_generation_prompt=True)


def complete(messages, max_tokens=256):
    """一次性生成(非流式)。"""
    from mlx_lm import generate
    s = _load()
    return generate(s["model"], s["tok"], prompt=_to_prompt(messages),
                    max_tokens=max_tokens, verbose=False).strip()


def stream(messages, max_tokens=256):
    """流式生成：逐块 yield(前端打字机效果)。"""
    from mlx_lm import stream_generate
    s = _load()
    for chunk in stream_generate(s["model"], s["tok"], prompt=_to_prompt(messages),
                                 max_tokens=max_tokens):
        yield chunk.text


# ==============================================================================
# ① 聊天 Web：Gradio ChatInterface(流式)。Gradio 6：history 默认就是 messages 字典列表
# ==============================================================================
def chat_fn(message, history):
    messages = (history or []) + [{"role": "user", "content": message}]
    acc = ""
    for piece in stream(messages, max_tokens=256):
        acc += piece
        yield acc                                            # 边生成边刷新


def build_demo():
    return gr.ChatInterface(
        fn=chat_fn, title="本地 LLM 聊天(mlx-lm)",
        description=f"模型：{MODEL_ID}，跑在 Mac(mps) 上，流式输出。API 见 `python3 app.py api`。",
        examples=["用一句话介绍机器学习", "写一句关于秋天的诗"],
        analytics_enabled=False,
    )


# ==============================================================================
# ② OpenAI 兼容 API：/v1/chat/completions（非流式 + 流式 SSE）
# ==============================================================================
def make_api():
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="本地 LLM 服务(OpenAI 兼容)")

    class ChatReq(BaseModel):
        model: str = MODEL_ID
        messages: list
        max_tokens: int = 256
        stream: bool = False

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_ID}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatReq):
        if req.stream:
            def sse():
                for piece in stream(req.messages, req.max_tokens):
                    chunk = {"choices": [{"delta": {"content": piece}}]}
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")
        text = complete(req.messages, req.max_tokens)
        return {"id": "chatcmpl-local", "object": "chat.completion", "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}]}

    return app


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if arg == "smoke":
        out = complete([{"role": "user", "content": "用一句话介绍机器学习"}], max_tokens=40)
        print("非流式:", out[:80])
        assert len(out) > 5
        acc = "".join(list(stream([{"role": "user", "content": "数到3"}], max_tokens=20)))
        print("流式拼接:", acc[:60])
        assert len(acc) > 0
        build_demo()
        print("✅ 项目3 自检通过：mlx-lm 加载 + 非流式/流式生成 + ChatInterface 构建。")
    elif arg == "api":
        import uvicorn
        uvicorn.run(make_api(), host="0.0.0.0", port=8000)   # OpenAI 兼容：POST /v1/chat/completions
    else:
        build_demo().queue().launch()
