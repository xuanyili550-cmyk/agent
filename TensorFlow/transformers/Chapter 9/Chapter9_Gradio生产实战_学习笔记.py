"""
================================================================================
 Chapter 9 · Gradio 生产实战 —— 系统学习笔记（Gradio 6 API，可运行自检）
================================================================================
 配套 HF 课程「Building and sharing demos」，但**全部用生产写法、不写玩具版**：
 真实模型、Blocks 布局、流式输出、并发队列、错误处理、示例缓存、鉴权、
 FastAPI 挂载、部署(Spaces/Docker)。本机 Gradio 6.26。

 ★ Gradio App 天生是“启动一个网页服务”，不像前几章能 print 完就退出。
   所以本文件的运行方式是「构建 + 自检」而不是「启动」：
     python3 Chapter9_Gradio生产实战_学习笔记.py          # 构建所有 demo + 直接调用底层函数自检(不起服务)
     python3 Chapter9_Gradio生产实战_学习笔记.py serve 2   # 真正启动第 2 个 demo(浏览器打开 http://127.0.0.1:7860)
   想看真界面就用 serve；平时跑默认模式验证代码正确即可。

 目录：
   1  Interface vs Blocks vs ChatInterface：三种入口怎么选
   2  生产级情感分析服务(Interface + 示例缓存 + 错误处理 + 队列)
   3  Blocks 布局：Row/Column/Tabs/Accordion + gr.State 会话状态
   4  流式输出(生成器 yield)：LLM/长任务边算边显示 + gr.Progress 进度
   5  ChatInterface：现代聊天 UI(流式 + system prompt 作为 additional_inputs)
   6  并发与队列 queue()/concurrency_limit：扛住多人同时用
   7  鉴权 auth / 环境变量 / 隐藏 API / analytics 关闭
   8  用 API 方式调用(api_name) + gr.mount_gradio_app 挂到 FastAPI
   9  部署：launch 参数、Hugging Face Spaces、Docker
================================================================================
"""

import sys

# ==============================================================================
# 1) 三种入口怎么选（生产项目 90% 用 Blocks 或 ChatInterface）
# ==============================================================================
# gr.Interface        —— 最快：给 fn + inputs + outputs，自动生成表单式界面。适合单一功能 demo。
# gr.Blocks           —— 最灵活：自己摆 Row/Column/Tabs、绑定多个按钮事件、共享 State。生产首选。
# gr.ChatInterface    —— 聊天专用：内置消息气泡/流式/重试/多轮历史，做对话应用直接用它。


# ==============================================================================
# 2) 生产级情感分析服务
# ==============================================================================
# 生产要点：①真实模型只加载一次(模块级/lru_cache) ②输入校验 + gr.Error 友好报错
#          ③examples + cache_examples 首屏就有结果 ④queue() 扛并发 ⑤flagging 收集坏样本
import gradio as gr

_clf = None
def _get_classifier():
    global _clf
    if _clf is None:
        from transformers import pipeline
        _clf = pipeline("sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english")
    return _clf


def analyze_sentiment(text: str):
    # 输入校验：空输入直接抛 gr.Error → 前端弹红条，而不是 500 崩掉
    if not text or not text.strip():
        raise gr.Error("请输入文本(不能为空)")
    if len(text) > 2000:
        raise gr.Error("文本过长(>2000 字符)，请缩短")
    r = _get_classifier()(text[:2000])[0]
    # gr.Label 组件要 {标签: 概率} 字典
    return {r["label"]: float(r["score"]), "OTHER": 1 - float(r["score"])}


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


# ==============================================================================
# 3) Blocks 布局 + gr.State 会话状态
# ==============================================================================
# gr.State：每个用户会话独立的“记忆”(如聊天历史/累计计数)，多次点击间保持。
def build_blocks_counter():
    def add(n, state):
        state = (state or 0) + n
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


# ==============================================================================
# 4) 流式输出(生成器) + gr.Progress 进度条
# ==============================================================================
# 长任务/LLM 别等全算完才返回：函数用 yield 逐步产出，界面边算边刷新，体验好得多。
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


# ==============================================================================
# 5) ChatInterface：生产级聊天 UI（流式 + 可调系统提示）
# ==============================================================================
# type="messages"：history 是 [{"role":"user/assistant","content":...}] 列表(OpenAI 格式)。
# 这里用一个“回声+规则”后端占位；真实项目把 respond 换成调你的 LLM(HF pipeline / API)。
def respond(message, history, system_prompt):
    # 真实写法：把 system_prompt + history + message 拼成对话，调 LLM 流式返回
    reply = f"[{(system_prompt or '助手').strip()}] 收到：{message}"
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


# ==============================================================================
# 6) 并发与队列（生产必开）
# ==============================================================================
# demo.queue()：开请求队列，避免多人同时用时挤爆。concurrency_limit 控制同时跑几个。
#   demo.queue(default_concurrency_limit=4, max_size=64)
#   .launch(max_threads=40)
# 重活(模型推理)务必开 queue，否则并发请求会阻塞/超时。


# ==============================================================================
# 7) 鉴权 / 环境 / 隐藏
# ==============================================================================
# launch() 层(v6 实测可用的参数)：
#   demo.launch(
#       auth=("admin", "secret"),                 # 简单账号密码；或 auth=函数(user,pwd)->bool
#       auth_message="内部工具，请登录",
#       server_name="0.0.0.0", server_port=7860,  # 对外暴露(容器里用 0.0.0.0)
#       share=False,                              # True 会生成 *.gradio.live 公网临时链接
#       ssr_mode=False,
#   )
# 构造器层：gr.Blocks(analytics_enabled=False) / gr.Interface(analytics_enabled=False) 关匿名统计。
# 隐藏 API：环境变量 GRADIO_SHOW_API=False，或给事件设 api_name=False(逐个隐藏)。
# 敏感信息(模型路径/密钥)走环境变量 os.environ，别硬编码。


# ==============================================================================
# 8) 用 API 方式调用 + 挂到 FastAPI
# ==============================================================================
# 每个事件可 api_name="predict"，外部就能用 gradio_client 或 HTTP 调：
#   from gradio_client import Client
#   Client("http://127.0.0.1:7860").predict("great!", api_name="/predict")
# 挂到已有 FastAPI 服务(和你自己的业务接口共存)：
#   from fastapi import FastAPI
#   app = FastAPI()
#   @app.get("/health")
#   def health(): return {"ok": True}
#   app = gr.mount_gradio_app(app, build_sentiment_app().queue(), path="/gradio")
#   # uvicorn 启动：uvicorn thismodule:app --host 0.0.0.0 --port 8000
#   # 业务接口在 /health，Gradio UI 在 /gradio —— 生产上常这么一体部署。


# ==============================================================================
# 9) 部署
# ==============================================================================
# · Hugging Face Spaces：仓库放 app.py + requirements.txt，Space 自动起服务(最省事)。
# · Docker：FROM python:3.11-slim → pip install → EXPOSE 7860 →
#           CMD ["python","app.py"]  (app 里 launch(server_name="0.0.0.0"))。
# · share=True 只适合临时演示(链接 72 小时失效、流量走 gradio 中转)，别用于正式生产。


# ==============================================================================
# 自检 / 启动
# ==============================================================================
BUILDERS = {
    1: ("情感分析服务", build_sentiment_app),
    2: ("Blocks+State 累加器", build_blocks_counter),
    3: ("流式输出+进度", build_stream_app),
    4: ("聊天机器人", build_chat_app),
}


def smoke():
    """构建所有 demo + 直接调用底层函数自检(不启动服务)。"""
    print("[自检] 逐个构建 demo 对象(验证组件图合法)...")
    for i, (name, b) in BUILDERS.items():
        demo = b()
        assert demo is not None
        print(f"   ✅ demo {i} 「{name}」构建成功 ({type(demo).__name__})")
    print("[自检] 直接调用底层函数(验证业务逻辑)...")
    r = analyze_sentiment("I love this!")
    assert max(r, key=r.get) == "POSITIVE", r
    print("   ✅ analyze_sentiment('I love this!') →", max(r, key=r.get))
    try:
        analyze_sentiment("   ")
        raise SystemExit("空输入应报错却没报")
    except gr.Error:
        print("   ✅ 空输入正确抛 gr.Error(前端会弹友好红条)")
    stream_out = list(slow_stream("a b c"))
    assert stream_out[-1].strip() == "a b c"
    print("   ✅ slow_stream 流式逐步产出，末帧 =", repr(stream_out[-1].strip()))
    chat_out = list(respond("hi", [], "机器人"))
    assert chat_out[-1].endswith("收到：hi")
    print("   ✅ respond 聊天流式产出，末帧 =", repr(chat_out[-1]))
    print("\n✅ 全部自检通过。想看真界面：python3", sys.argv[0].split('/')[-1], "serve 1")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        idx = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        name, builder = BUILDERS[idx]
        print(f"启动「{name}」→ http://127.0.0.1:7860")
        builder().queue().launch()
    else:
        smoke()
