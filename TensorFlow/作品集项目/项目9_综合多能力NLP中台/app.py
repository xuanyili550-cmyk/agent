"""
作品集项目9 · 多能力 NLP 中台（综合,生产级,可部署 HF Spaces）
--------------------------------------------------------------------------------
 一句话：情感/零样本分类/NER/完形填空 四项能力集成一个 Tab 网页 + 统一 REST API。
 生产要点：各能力模型分别懒加载单例、输入校验、Tab UI、FastAPI 统一出 API。
 注：transformers v5 已移除 summarization/translation/QA pipeline,本中台只用仍在的任务。
 运行：python3 app.py [smoke|api]
"""
import sys
import gradio as gr

MODELS = {
    "sentiment": "uer/roberta-base-finetuned-dianping-chinese",   # 中文情感
    "ner": "dslim/bert-base-NER",
    "zeroshot": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "fillmask": "bert-base-chinese",
}
_PIPES = {}


def _pipe(task):
    if task not in _PIPES:
        from transformers import pipeline
        name = {"sentiment": "text-classification", "ner": "token-classification",
                "zeroshot": "zero-shot-classification", "fillmask": "fill-mask"}[task]
        kw = {"aggregation_strategy": "simple"} if task == "ner" else {}
        _PIPES[task] = pipeline(name, model=MODELS[task], **kw)
    return _PIPES[task]


def do_sentiment(text):
    if not (text or "").strip():
        raise gr.Error("请输入文本")
    return {d["label"]: float(d["score"]) for d in _pipe("sentiment")(text)}


def do_ner(text):
    if not (text or "").strip():
        raise gr.Error("请输入文本")
    return "  ".join(f"{e['word']}({e['entity_group']})" for e in _pipe("ner")(text)) or "未识别到实体"


def do_zeroshot(text, labels):
    cand = [x.strip() for x in (labels or "").split(",") if x.strip()]
    if not (text or "").strip() or len(cand) < 2:
        raise gr.Error("需要文本 + 至少 2 个标签")
    r = _pipe("zeroshot")(text, candidate_labels=cand)
    return {l: float(s) for l, s in zip(r["labels"], r["scores"])}


def do_fillmask(text):
    if "[MASK]" not in (text or ""):
        raise gr.Error("文本里要含 [MASK]")
    return {d["token_str"]: float(d["score"]) for d in _pipe("fillmask")(text)[:5]}


def build_ui():
    with gr.Blocks(title="多能力 NLP 中台") as demo:
        gr.Markdown("# 🧰 多能力 NLP 中台\n情感 / 零样本分类 / NER / 完形填空")
        with gr.Tab("情感分析"):
            t1 = gr.Textbox(label="中文文本", value="这家店服务态度很好")
            gr.Button("分析").click(do_sentiment, t1, gr.Label())
        with gr.Tab("零样本分类"):
            t2 = gr.Textbox(label="文本", value="苹果发布了新手机")
            l2 = gr.Textbox(label="标签(逗号)", value="科技,体育,财经")
            gr.Button("分类").click(do_zeroshot, [t2, l2], gr.Label())
        with gr.Tab("命名实体"):
            t3 = gr.Textbox(label="文本(英文模型)", value="My name is Wolfgang and I live in Berlin")
            gr.Button("抽取").click(do_ner, t3, gr.Textbox(label="实体"))
        with gr.Tab("完形填空"):
            t4 = gr.Textbox(label="含 [MASK] 的文本", value="今天天气真[MASK]。")
            gr.Button("预测").click(do_fillmask, t4, gr.Label())
    return demo


def smoke():
    import inspect
    # 校验各能力的输入校验分支(不下模型)
    checks = [(do_sentiment, ("",)), (do_ner, ("",)), (do_zeroshot, ("", "a")), (do_fillmask, ("无mask",))]
    for fn, args in checks:
        try:
            fn(*args); raise SystemExit(f"{fn.__name__} 应抛 gr.Error")
        except gr.Error:
            pass
    build_ui()
    assert all(t in MODELS for t in ("sentiment", "ner", "zeroshot", "fillmask"))
    print("✅ 项目9 自检通过:4 能力输入校验 + Tab UI 构建 OK(真实推理需下各模型)")


def mount_api():
    from fastapi import FastAPI
    app = FastAPI(title="多能力 NLP 中台")
    app.get("/api/health")(lambda: {"status": "ok", "models": MODELS})
    app.post("/api/sentiment")(lambda text: do_sentiment(text))
    app.post("/api/ner")(lambda text: {"entities": do_ner(text)})
    app.post("/api/zeroshot")(lambda text, labels: do_zeroshot(text, labels))
    app.post("/api/fillmask")(lambda text: do_fillmask(text))
    return gr.mount_gradio_app(app, build_ui(), path="/")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "smoke":
        smoke()
    elif mode == "api":
        import uvicorn; uvicorn.run(mount_api(), host="0.0.0.0", port=7860)
    else:
        build_ui().launch()
