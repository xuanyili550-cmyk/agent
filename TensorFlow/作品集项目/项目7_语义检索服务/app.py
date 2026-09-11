"""
作品集项目7 · 语义检索服务（纯 HF 底层,生产级,可部署 HF Spaces）
--------------------------------------------------------------------------------
 一句话：文档建向量库 → 按意思检索。只用 AutoTokenizer+AutoModel 手写(分词→前向→池化→归一化→余弦)。
 生产要点：模型单例、输入校验(gr.Error)、mask 加权 mean 池化、L2 归一化(点积=余弦)、FastAPI 出 API。
 运行：
   python3 app.py smoke   # 离线自检(校验输入+构建UI,不下模型)
   python3 app.py         # 起 Gradio 网页
   python3 app.py api     # 起 FastAPI(网页/ + API /api/search)
"""
import sys
import numpy as np
import gradio as gr

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_STATE = {}


def _load():
    """模型单例:首次调用才加载(生产避免每请求重载)。"""
    if "model" not in _STATE:
        import torch
        from transformers import AutoModel, AutoTokenizer
        _STATE["dev"] = "mps" if torch.backends.mps.is_available() else "cpu"
        _STATE["tok"] = AutoTokenizer.from_pretrained(MODEL)
        _STATE["model"] = AutoModel.from_pretrained(MODEL).to(_STATE["dev"]).eval()
    return _STATE


def _embed(texts):
    """纯底层:分词(mask)→AutoModel→mask 加权 mean 池化→L2 归一化。"""
    import torch
    s = _load()
    enc = s["tok"](texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(s["dev"])
    with torch.inference_mode():
        hidden = s["model"](**enc).last_hidden_state            # [B,L,H]
    mask = enc["attention_mask"].unsqueeze(-1).float()
    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean 池化(排除 PAD)
    vec = vec / vec.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # L2 归一化
    return vec.cpu().numpy()


def search(docs_text: str, query: str, top_k: int = 3) -> str:
    docs = [d.strip() for d in (docs_text or "").splitlines() if d.strip()]
    if not docs:
        raise gr.Error("请在左侧每行输入一条文档")
    if not (query or "").strip():
        raise gr.Error("请输入查询")
    doc_vecs = _embed(docs)
    q_vec = _embed([query])[0]
    sims = doc_vecs @ q_vec                                     # 归一化后点积=余弦
    order = np.argsort(-sims)[:int(top_k)]
    return "\n".join(f"{sims[i]:.3f}  {docs[i]}" for i in order)


def build_ui():
    with gr.Blocks(title="语义检索服务") as demo:
        gr.Markdown("# 🔍 语义检索服务（纯 HF 底层）\n每行一条文档,按意思检索最相关的。")
        with gr.Row():
            docs = gr.Textbox(label="文档库(每行一条)", lines=8,
                              value="退货政策是7天无理由\n满99元包邮\n客服工作时间9点到18点")
            with gr.Column():
                q = gr.Textbox(label="查询", value="怎么退货")
                k = gr.Slider(1, 5, value=3, step=1, label="返回条数")
                btn = gr.Button("检索", variant="primary")
        out = gr.Textbox(label="结果(相似度 + 文档)", lines=6)
        btn.click(search, [docs, q, k], out)
    return demo


def smoke():
    # 离线自检:只校验输入 + 构建 UI,不加载模型(不联网)
    for bad in [("", "q"), ("d", "")]:
        try:
            search(*bad)
            raise SystemExit("应抛 gr.Error 却没抛")
        except gr.Error:
            pass
    build_ui()
    print("✅ 项目7 自检通过:输入校验 + UI 构建 OK(真实检索需下 all-MiniLM 模型)")


def mount_api():
    from fastapi import FastAPI
    app = FastAPI(title="语义检索服务")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model": MODEL}

    @app.post("/api/search")
    def api_search(docs: str, query: str, top_k: int = 3):
        return {"results": search(docs, query, top_k)}

    return gr.mount_gradio_app(app, build_ui(), path="/")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "smoke":
        smoke()
    elif mode == "api":
        import uvicorn
        uvicorn.run(mount_api(), host="0.0.0.0", port=7860)
    else:
        build_ui().launch()
