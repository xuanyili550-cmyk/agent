"""
================================================================================
 Chapter 9 案例闯关 · Gradio 生产实战（4 个完整可上线的 App，非玩具版）
================================================================================
 全部 Gradio 6 写法、真实模型、生产结构(布局/错误处理/队列/流式/鉴权/FastAPI)。

 用法：
   python3 Chapter9_案例闯关_Gradio生产实战.py            # 自检：构建4个App+调用底层函数(不起服务)
   python3 Chapter9_案例闯关_Gradio生产实战.py serve 1     # 启动第1个App(浏览器开 127.0.0.1:7860)
   python3 Chapter9_案例闯关_Gradio生产实战.py serve 4     # 第4个用 uvicorn 起(FastAPI 一体化)

 四个 App：
   1  多模型文本工具台   Blocks+Tabs：情感分析 / 摘要 / 命名实体识别(3 个真实模型)
   2  流式聊天机器人     ChatInterface + TextIteratorStreamer 真流式 + 可调采样参数
   3  语义检索服务       句向量嵌入 + 余弦检索(FAQ 知识库检索 UI，RAG 检索前半段)
   4  FastAPI 一体化     业务 JSON API(/predict、/health) + Gradio UI(/gradio) + 鉴权

 说明：用的都是小/已缓存模型(distilbert-sst2 / t5-small / bert-finetuned-ner /
 distilgpt2 / MiniLM)。首次下载后走缓存。App 天生要起 web 服务，所以默认是“构建+自检”。
================================================================================
"""

import sys
import gradio as gr

# --- 模型惰性单例(生产必须：加载一次、全程复用) ------------------------------
_cache = {}
def _pipe(task, model):
    key = (task, model)
    if key not in _cache:
        from transformers import pipeline
        _cache[key] = pipeline(task, model=model)
    return _cache[key]


# ==============================================================================
# App 1 · 多模型文本工具台（Blocks + Tabs，3 个真实任务）
# ==============================================================================
def app1_text_toolbox():
    def do_sentiment(text):
        if not text.strip():
            raise gr.Error("请输入文本")
        r = _pipe("sentiment-analysis",
                  "distilbert-base-uncased-finetuned-sst-2-english")(text[:1000])[0]
        return {r["label"]: float(r["score"])}

    def do_summary(text):
        if len(text.split()) < 20:
            raise gr.Error("文本太短，摘要需要至少 ~20 词")
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        if "t5" not in _cache:
            tok = AutoTokenizer.from_pretrained("t5-small")
            _cache["t5"] = (tok, AutoModelForSeq2SeqLM.from_pretrained("t5-small").eval())
        tok, model = _cache["t5"]
        ids = tok("summarize: " + text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model.generate(**ids, max_length=60, min_length=10, num_beams=4)
        return tok.decode(out[0], skip_special_tokens=True)

    def do_ner(text):
        if not text.strip():
            raise gr.Error("请输入文本")
        if "ner" not in _cache:
            from transformers import pipeline
            # aggregation_strategy="simple"：把子词合并成实体，返回带 entity_group 的结果
            _cache["ner"] = pipeline("token-classification",
                                     model="huggingface-course/bert-finetuned-ner",
                                     aggregation_strategy="simple")
        ents = _cache["ner"](text)
        # 用 gr.HighlightedText 展示：把实体片段高亮(生产里常这么可视化 NER)
        spans, last = [], 0
        for e in ents:
            if e["start"] > last:
                spans.append((text[last:e["start"]], None))
            spans.append((text[e["start"]:e["end"]], e["entity_group"]))
            last = e["end"]
        spans.append((text[last:], None))
        return spans

    with gr.Blocks(title="文本工具台", analytics_enabled=False) as demo:
        gr.Markdown("# 🧰 多模型文本工具台\n三个真实模型，Tabs 切换。生产结构：单例模型 + 错误处理。")
        with gr.Tabs():
            with gr.TabItem("情感分析"):
                t1 = gr.Textbox(label="文本", lines=3)
                o1 = gr.Label(label="情感")
                gr.Button("分析", variant="primary").click(do_sentiment, t1, o1)
                gr.Examples([["I really love this product!"]], t1)
            with gr.TabItem("摘要"):
                t2 = gr.Textbox(label="长文本", lines=6)
                o2 = gr.Textbox(label="摘要", interactive=False)
                gr.Button("摘要", variant="primary").click(do_summary, t2, o2)
            with gr.TabItem("命名实体识别"):
                t3 = gr.Textbox(label="文本", lines=3)
                o3 = gr.HighlightedText(label="实体")
                gr.Button("识别", variant="primary").click(do_ner, t3, o3)
                gr.Examples([["My name is Sylvain and I work at Hugging Face in Brooklyn."]], t3)
    return demo, {"sentiment": do_sentiment, "summary": do_summary, "ner": do_ner}


# ==============================================================================
# App 2 · 流式聊天机器人（ChatInterface + 真 TextIteratorStreamer）
# ==============================================================================
def app2_streaming_chat():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    import threading

    ckpt = "distilgpt2"     # 生产里换成真 instruct 模型(Qwen/Llama)；这里用小的演示流式
    if "chat" not in _cache:
        tok = AutoTokenizer.from_pretrained(ckpt)
        tok.pad_token = tok.eos_token
        _cache["chat"] = (tok, AutoModelForCausalLM.from_pretrained(ckpt).eval())
    tok, model = _cache["chat"]

    def respond(message, history, max_new_tokens, temperature):
        # 把多轮历史 + 本轮拼成 prompt(真 instruct 模型应用其 chat template)
        prompt = ""
        for turn in (history or []):
            prompt += turn["content"] + "\n"
        prompt += message + "\n"
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**inputs, streamer=streamer, max_new_tokens=int(max_new_tokens),
                      do_sample=True, temperature=float(temperature), top_p=0.95)
        threading.Thread(target=model.generate, kwargs=kwargs).start()
        acc = ""
        for token in streamer:          # ← 真流式：模型边生成，UI 边刷新
            acc += token
            yield acc

    demo = gr.ChatInterface(
        fn=respond,
        title="💬 流式聊天机器人",
        description="真 TextIteratorStreamer 流式；滑杆调采样参数(生产里换成真 LLM)。",
        additional_inputs=[
            gr.Slider(8, 128, value=40, step=8, label="max_new_tokens"),
            gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="temperature"),
        ],
    )
    return demo, {"respond": respond}


# ==============================================================================
# App 3 · 语义检索服务（嵌入 + 余弦，FAQ 知识库检索 UI）
# ==============================================================================
def app3_semantic_search():
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    FAQ = [
        ("怎么重置密码？", "去 设置 > 安全 > 重置密码，按邮件链接操作。"),
        ("App 上传照片会闪退", "请升级到 v3.2 以上版本，该崩溃已修复。"),
        ("我被重复扣款了", "核实后 3-5 个工作日内退还重复扣款。"),
        ("怎么联系人工", "在聊天里输入 'agent'，或工作日 9-18 点拨打热线。"),
    ]
    # ★中文 FAQ 必须用多语言嵌入模型；用英文的 all-MiniLM 检索中文会失准(生产常见坑)
    ck = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    if "emb" not in _cache:
        tok = AutoTokenizer.from_pretrained(ck)
        _cache["emb"] = (tok, AutoModel.from_pretrained(ck).eval())
    tok, model = _cache["emb"]

    def embed(texts):
        enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(v, p=2, dim=1)

    faq_vecs = embed([q for q, _ in FAQ])

    def search(query, top_k):
        if not query.strip():
            raise gr.Error("请输入问题")
        sims = (embed([query]) @ faq_vecs.T)[0]
        idx = sims.argsort(descending=True)[:int(top_k)]
        return "\n\n".join(
            f"[{sims[i]:.2f}] {FAQ[i][0]}\n  → {FAQ[i][1]}" for i in idx)

    with gr.Blocks(title="语义检索", analytics_enabled=False) as demo:
        gr.Markdown("# 🔎 语义检索服务\n按“意思”检索 FAQ(换个说法也能命中)，这就是 RAG 的检索前半段。")
        q = gr.Textbox(label="用户问题", placeholder="照片传不上去还闪退")
        k = gr.Slider(1, 4, value=2, step=1, label="返回条数")
        out = gr.Textbox(label="命中的 FAQ", lines=6, interactive=False)
        gr.Button("检索", variant="primary").click(search, [q, k], out)
        gr.Examples([["照片传不上去还老是崩"], ["忘记登录密码怎么办"]], q)
    return demo, {"search": search}


# ==============================================================================
# App 4 · FastAPI 一体化（业务 JSON API + Gradio UI + 鉴权）
# ==============================================================================
def app4_fastapi_mount():
    from fastapi import FastAPI

    api = FastAPI(title="生产一体化服务")

    @api.get("/health")                      # 业务健康检查(K8s 探针常用)
    def health():
        return {"status": "ok"}

    @api.get("/predict")                     # 业务 JSON API(给程序调，不走 UI)
    def predict(text: str):
        r = _pipe("sentiment-analysis",
                  "distilbert-base-uncased-finetuned-sst-2-english")(text[:1000])[0]
        return {"label": r["label"], "score": round(float(r["score"]), 4)}

    ui, _ = app3_semantic_search()           # 复用 App3 当 UI
    # 把 Gradio UI 挂到 /gradio；鉴权可加 auth=... ；同一个进程既有 REST 又有 UI
    api = gr.mount_gradio_app(api, ui.queue(), path="/gradio")
    return api, {"predict": predict, "health": health}


# ==============================================================================
# 菜单 / 自检
# ==============================================================================
APPS = {
    1: ("多模型文本工具台", app1_text_toolbox),
    2: ("流式聊天机器人", app2_streaming_chat),
    3: ("语义检索服务", app3_semantic_search),
    4: ("FastAPI 一体化", app4_fastapi_mount),
}


def smoke():
    print("[自检] 构建 4 个 App + 调用底层函数(不起服务)...\n")
    d1, f1 = app1_text_toolbox()
    assert max(f1["sentiment"]("I love it!"), key=f1["sentiment"]("I love it!").get) == "POSITIVE"
    ner = f1["ner"]("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    assert any(tag == "PER" for _, tag in ner)
    print("  ✅ App1 工具台：情感=POSITIVE、NER 识别出 PER 实体、Blocks 构建成功")

    d2, f2 = app2_streaming_chat()
    frames = list(f2["respond"]("Hello", [], 8, 0.8))
    assert len(frames) >= 1 and len(frames[-1]) >= len(frames[0])
    print(f"  ✅ App2 流式聊天：真流式产出 {len(frames)} 帧(边生成边刷新)、ChatInterface 构建成功")

    d3, f3 = app3_semantic_search()
    res = f3["search"]("照片传不上去还老是崩", 1)
    assert "闪退" in res or "崩溃" in res
    print("  ✅ App3 语义检索：换个说法也命中“上传照片闪退”那条 FAQ")

    api, f4 = app4_fastapi_mount()
    assert f4["health"]()["status"] == "ok"
    assert f4["predict"]("great!")["label"] == "POSITIVE"
    assert any(getattr(r, "path", "") == "/gradio" for r in api.routes)
    print("  ✅ App4 FastAPI：/health 正常、/predict=POSITIVE、Gradio 已挂到 /gradio")
    print("\n✅ 4 个生产 App 全部自检通过。启动看真界面：python3", sys.argv[0].split('/')[-1], "serve 1")


def serve(idx):
    name, builder = APPS[idx]
    built = builder()
    if idx == 4:
        import uvicorn
        print(f"启动「{name}」→ REST: http://127.0.0.1:8000/health  UI: http://127.0.0.1:8000/gradio")
        uvicorn.run(built[0], host="127.0.0.1", port=8000)
    else:
        demo = built[0]
        print(f"启动「{name}」→ http://127.0.0.1:7860")
        demo.queue().launch()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        serve(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    else:
        smoke()
