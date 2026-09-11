"""
作品集项目8 · 零样本文本分类（生产级,可部署 HF Spaces）
--------------------------------------------------------------------------------
 一句话：不训练,给文本 + 自定义候选标签,NLI 模型直接判类别(标签可随时改)。
 生产要点：模型单例、输入校验、候选标签运行时传入(零样本)、Web + FastAPI API。
 运行：python3 app.py [smoke|api]
"""
import sys
import gradio as gr

MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"   # 多语言 NLI,中英文可用
_STATE = {}


def _load():
    if "clf" not in _STATE:
        from transformers import pipeline
        _STATE["clf"] = pipeline("zero-shot-classification", model=MODEL)
    return _STATE["clf"]


def classify(text: str, labels: str) -> dict:
    text = (text or "").strip()
    cand = [x.strip() for x in (labels or "").split(",") if x.strip()]
    if not text:
        raise gr.Error("请输入文本")
    if len(cand) < 2:
        raise gr.Error("请至少用逗号分隔 2 个候选标签")
    r = _load()(text, candidate_labels=cand)
    return {lab: float(sc) for lab, sc in zip(r["labels"], r["scores"])}


def build_ui():
    with gr.Blocks(title="零样本文本分类") as demo:
        gr.Markdown("# 🏷️ 零样本文本分类\n给文本 + 候选标签(逗号分隔),不训练直接分类。")
        text = gr.Textbox(label="文本", value="这家餐厅的菜又贵又难吃")
        labels = gr.Textbox(label="候选标签(逗号分隔)", value="餐饮,科技,体育,金融")
        btn = gr.Button("分类", variant="primary")
        out = gr.Label(label="各标签置信度")
        btn.click(classify, [text, labels], out)
    return demo


def smoke():
    for bad in [("", "a,b"), ("x", "onlyone")]:
        try:
            classify(*bad); raise SystemExit("应抛 gr.Error")
        except gr.Error:
            pass
    build_ui()
    print("✅ 项目8 自检通过:输入校验 + UI 构建 OK(真实分类需下 NLI 模型)")


def mount_api():
    from fastapi import FastAPI
    app = FastAPI(title="零样本分类")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model": MODEL}

    @app.post("/api/classify")
    def api_classify(text: str, labels: str):
        return classify(text, labels)

    return gr.mount_gradio_app(app, build_ui(), path="/")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "smoke":
        smoke()
    elif mode == "api":
        import uvicorn; uvicorn.run(mount_api(), host="0.0.0.0", port=7860)
    else:
        build_ui().launch()
