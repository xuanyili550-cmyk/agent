import  sys
import gradio as gr
_clf=None
def _get_classifier():
    global  _clf
    if _clf is None:
        from transformers import pipeline
        _clf=pipeline("sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english")
    return _clf

def analyze_sentiment(text:str):
    if not text or not text.strip():
        raise gr.Error("请输入文本(不能为空)")
    if len(text) > 2000:
        raise gr.Error("文本过长(>2000 字符)，请缩短")
    r=_get_classifier()(text[:2000])[0]
    return {r['label']:float (r["score"]),"OTHER": 1 - float(r["score"])}

def build_sentiment_app():
    return gr.Interface(
        fn=analyze_sentiment,
        inputs=gr.Textbox(label="输入评论", lines=3, placeholder="This movie is..."),
        outputs=gr.Label(num_top_classes=2, label="情感"),
        title="情感分析服务",
        description="生产示例：输入校验 + 错误处理 + 示例缓存 + 并发队列。",
        examples=[["I absolutely loved this!"], ["This is the worst thing ever."]],
        cache_examples=False,           # 生产可设 True：预跑示例，首屏秒出(启动稍慢)
        flagging_mode="manual",         # 用户可“标记”坏结果，落盘到 flagged/ 供后续排查
    )

def build_blocks_counter():
    def add(n,state):
        state=(state or 0)+n
        return state, f"累计：{state}"
    with gr.Blocks(title="Blocks + State") as demo:
        gr.Markdown("## Blocks 布局演示：累加器(演示会话状态 gr.State)")
        st = gr.State(0)                       # 每个访客独立的状态
        with gr.Row():                          # 一行里并排放
            num = gr.Number(value=1, label="加多少")
            btn = gr.Button("累加", variant="primary")
        out = gr.Textbox(label="结果", interactive=False)
        btn.click(add, inputs=[num, st], outputs=[st, out])
    return demo
def slow_stream(text, progress=gr.Progress()):
    words = (text or "hello world from gradio").split()
    acc = ""
    for i, w in enumerate(progress.tqdm(words, desc="生成中")):
        acc += w + " "
        yield acc                          # 每 yield 一次，前端就更新一次(流式)


def build_stream_app():
    return gr.Interface(fn=slow_stream, inputs="text", outputs="text",
                        title="流式输出 + 进度条",
                        description="生成器 yield 实现边算边显示；gr.Progress 显示进度。")
def respond(message, history, system_prompt):
    # 真实写法：把 system_prompt + history + message 拼成对话，调 LLM 流式返回
    reply = f"[{(system_prompt or  history).strip()}] 收到：{message}"
    acc = ""
    for ch in reply:                        # 模拟流式：逐字吐出
        acc += ch
        yield acc


def build_chat_app():
    return gr.ChatInterface(
        fn=respond,                         # Gradio 6：history 默认就是 messages 字典列表
        title="生产级聊天机器人",
        description="流式回复 + 可调 system prompt(生产里把 respond 换成真 LLM)。",
        additional_inputs=[gr.Textbox("你是一个乐于助人的助手", label="System Prompt")],
    )