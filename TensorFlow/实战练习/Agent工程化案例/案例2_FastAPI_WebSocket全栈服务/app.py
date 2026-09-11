"""
================================================================================
 Agent 工程化案例2 · FastAPI + WebSocket 全栈 Agent 服务（对标 deepsearch 全栈落地）
================================================================================
 补齐"全栈可部署 + 流式进度"短板：Agent 跑在 FastAPI 后端，通过 WebSocket 把【每一步进度 + 逐字生成】
 实时推给前端(前端拿到就能做进度条/打字机)。配 Dockerfile + docker-compose.yml 一键部署。
 技术：FastAPI + WebSocket(实时进度) + mlx-lm 流式生成 + bge 本地检索。用 TestClient 本机真测 WebSocket。
 运行：
   python3 app.py smoke   # 用 TestClient 连 WebSocket、发问题、收流式进度+回答(本机真测，不起服务)
   python3 app.py         # 起服务(uvicorn)，WebSocket 在 ws://localhost:8000/ws/research
================================================================================
"""
import sys
import asyncio
import json

_M = {}
QP = "为这个句子生成表示以用于检索相关文章："
KB = ["RAG 先检索相关文档再让 LLM 基于文档回答，减少幻觉、可溯源。",
      "混合检索=向量+BM25 用 RRF 融合，召回更全。",
      "重排用 cross-encoder 对召回精排，精度更高。"]


def _dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _search(query, k=2):
    import torch
    import torch.nn.functional as F
    if "emb" not in _M:
        from transformers import AutoTokenizer, AutoModel
        _M["et"] = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
        _M["em"] = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5").to(_dev()).eval()

    def emb(ts, q=False):
        if q:
            ts = [QP + t for t in ts]
        enc = _M["et"](ts, padding=True, truncation=True, return_tensors="pt").to(_dev())
        with torch.no_grad():
            v = _M["em"](**enc).last_hidden_state[:, 0]
        return F.normalize(v, p=2, dim=1)
    if "kv" not in _M:
        _M["kv"] = emb(KB)
    sims = (emb([query], q=True) @ _M["kv"].T)[0]
    return [KB[int(i)] for i in sims.argsort(descending=True)[:k]]


def _mlx_stream(prompt, n=80):
    if "m" not in _M:
        from mlx_lm import load
        _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import stream_generate
    t = _M["t"].apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    for chunk in stream_generate(_M["m"], _M["t"], prompt=t, max_tokens=n):
        yield chunk.text


async def run_agent(query, emit):
    """跑 Agent，每一步/每个 token 通过 emit(异步回调)推给前端。"""
    await emit({"type": "progress", "step": "检索知识库…"})
    hits = _search(query, k=2)
    await emit({"type": "progress", "step": f"检索到 {len(hits)} 条资料"})
    await emit({"type": "sources", "data": hits})
    await emit({"type": "progress", "step": "生成回答(流式)…"})
    ctx = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(hits))
    acc = ""
    for piece in _mlx_stream(f"只根据资料回答。\n资料：\n{ctx}\n问题：{query}\n回答：", 80):
        acc += piece
        await emit({"type": "token", "data": piece})       # 逐字推(打字机)
        await asyncio.sleep(0)
    await emit({"type": "done", "answer": acc})
    return acc


def make_app():
    import os
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import FileResponse, HTMLResponse
    app = FastAPI(title="全栈 Agent 服务")
    INDEX = os.path.join(os.path.dirname(__file__), "static", "index.html")

    @app.get("/")                                            # 前端页面(真·全栈：后端+前端)
    def index():
        return FileResponse(INDEX) if os.path.exists(INDEX) else HTMLResponse("<h1>Agent 服务</h1>")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.websocket("/ws/research")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        query = (await websocket.receive_json()).get("query", "")

        async def emit(msg):
            await websocket.send_json(msg)
        try:
            await run_agent(query, emit)
        finally:
            await websocket.close()

    return app


def smoke():
    from fastapi.testclient import TestClient
    app = make_app()
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        page = client.get("/")                                    # 前端页面(真·全栈)
        assert page.status_code == 200 and "WebSocket" in page.text
        with client.websocket_connect("/ws/research") as ws:      # 本机真连 WebSocket
            ws.send_json({"query": "什么是混合检索"})
            msgs, answer = [], ""
            while True:
                m = ws.receive_json()
                msgs.append(m["type"])
                if m["type"] == "token":
                    answer += m["data"]
                if m["type"] == "done":
                    answer = m["answer"]; break
    # 自检噪音：期望消息类型序列含 progress/sources/token/done，流式回答非空（由下方 assert 保证）
    # print("  WebSocket 收到消息类型序列:", [t for t in ["progress", "sources", "token", "done"] if t in msgs])
    # print("  流式回答:", answer[:70])
    assert "progress" in msgs and "token" in msgs and "done" in msgs
    assert answer
    print("✅ 案例2 自检通过：FastAPI + WebSocket 实时进度/流式回答，TestClient 本机真测通过。")
    # print("  部署：docker compose up(见同目录 Dockerfile / docker-compose.yml)。前端 new WebSocket('ws://.../ws/research')。")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke()
    else:
        import uvicorn
        uvicorn.run(make_app(), host="0.0.0.0", port=8000)
