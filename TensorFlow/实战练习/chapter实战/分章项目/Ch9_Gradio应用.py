"""
================================================================================
 分章项目 · Ch9 · Gradio 应用（贴 HF Ch9：Interface + Blocks + ChatInterface）
================================================================================
 HF 课程 Ch9 讲“把模型包成能点的网页/API”，用可运行代码复现三种写法：
   ① Interface：最快的“一个 fn + inputs + outputs = 一个 App”，还自动生成 REST API。
   ② Blocks：低层灵活布局，自定义组件排布/多组件联动(生产界面常用)。
   ③ ChatInterface：一行做一个聊天机器人 UI(对话类应用)。
 为什么用 Gradio：ML 模型做出来要给人用/演示，Gradio 是最快的“模型→网页/API”方式(HF Spaces 一键部署)。
   完整章节材料见 ../../../Chapter 9/。
 跑：python3 Ch9_Gradio应用.py             # 起 Interface 网页 http://127.0.0.1:7860
    python3 Ch9_Gradio应用.py blocks      # 起 Blocks 版网页
    python3 Ch9_Gradio应用.py chat        # 起 ChatInterface 聊天网页
    python3 Ch9_Gradio应用.py smoke       # 只构建三种 UI + 调函数自检(不起服务)
================================================================================
"""
import sys
import gradio as gr

_clf = None
def get_clf():
    global _clf
    if _clf is None:
        from transformers import pipeline
        _clf = pipeline("sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english")
    return _clf


def analyze(text):
    if not text.strip():
        raise gr.Error("请输入文本")              # 输入校验：前端弹友好红条
    r = get_clf()(text[:1000])[0]
    return {r["label"]: float(r["score"]), "OTHER": 1 - float(r["score"])}


# ==============================================================================
# ① Interface：一个 fn + inputs + outputs
# ==============================================================================
def build_interface():
    return gr.Interface(
        fn=analyze,
        inputs=gr.Textbox(label="输入文本", lines=3, placeholder="This movie is..."),
        outputs=gr.Label(label="情感"),
        title="情感分析(Interface)",
        description="Ch9：一个 fn + inputs + outputs 就是一个 App，还自动带 REST API。",
        examples=[["I love this!"], ["This is terrible."]],
    )


# ==============================================================================
# ② Blocks：低层灵活布局(自定义组件排布 + 事件绑定)
# ==============================================================================
def build_blocks():
    with gr.Blocks(title="情感分析(Blocks)") as demo:
        gr.Markdown("## 情感分析 · Blocks 版\n用 Blocks 自定义布局：左输入右输出，按钮触发。")
        with gr.Row():
            inp = gr.Textbox(label="输入文本", lines=3)
            out = gr.Label(label="情感")
        btn = gr.Button("分析", variant="primary")
        btn.click(fn=analyze, inputs=inp, outputs=out)     # 事件绑定：点按钮→调 analyze
        gr.Examples([["Amazing product!"], ["Waste of money."]], inputs=inp)
    return demo


# ==============================================================================
# ③ ChatInterface：一行做聊天机器人 UI
# ==============================================================================
def chat_fn(message, history):
    # history 是过往对话；这里用情感模型演示“根据用户情绪回应”(生产接 LLM 生成回复)
    label, score = (lambda r: (r["label"], r["score"]))(get_clf()(message[:1000])[0])
    mood = "看起来你挺满意的 😊" if label == "POSITIVE" else "似乎你有些不满，我帮你转人工 🙋"
    return f"[情感={label} {score:.2f}] {mood}"


def build_chat():
    return gr.ChatInterface(fn=chat_fn, title="情感感知客服(ChatInterface)",
                            examples=["这个功能太好用了！", "又崩溃了，太差劲"])


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "interface"
    if arg == "smoke":
        # 三种 UI 都能构建 + 业务函数自检(不起服务)
        build_interface(); build_blocks(); build_chat()
        assert max(analyze("I love it!"), key=analyze("I love it!").get) == "POSITIVE"
        assert "转人工" in chat_fn("又崩溃了太差劲", [])
        print("✅ Interface / Blocks / ChatInterface 三种 UI 构建成功，函数自检通过。")
        # print("   起网页：python3 Ch9_Gradio应用.py [interface|blocks|chat]")
        # print("面试：Q Interface vs Blocks 怎么选? Q Gradio 怎么快速上线/自动出 API? (见 ../../../Chapter 9/)")
    elif arg == "blocks":
        build_blocks().launch()
    elif arg == "chat":
        build_chat().launch()
    else:
        build_interface().queue().launch()
