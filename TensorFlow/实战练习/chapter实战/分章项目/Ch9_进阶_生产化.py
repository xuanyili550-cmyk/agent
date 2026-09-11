"""
================================================================================
 分章项目 · Ch9 进阶 · Gradio 生产化（状态/进度/流式/挂FastAPI/鉴权）
================================================================================
 Ch9_Gradio应用.py 讲了三种入口(Interface/Blocks/ChatInterface)的基础；本文件补上
 HF Ch9 里“上生产”真正要用的能力，全部可构建自检：
   ① gr.State：多轮会话状态(每个用户独立)——聊天/计数器/购物车都靠它。
   ② 流式输出：函数用 yield 边算边吐(配合大模型生成，体验关键)。
   ③ gr.Progress：长任务进度条，让用户知道“在跑、跑到哪了”。
   ④ mount_gradio_app：把 Gradio UI 挂到 FastAPI，业务 API + 演示界面一体化部署。
   ⑤ 鉴权 auth / 关匿名统计 analytics_enabled——上线前的安全与合规开关。
   完整章节材料见 ../../../Chapter 9/。
 跑：python3 Ch9_进阶_生产化.py smoke     # 构建全部 UI + 调函数自检(不起服务，推荐)
    python3 Ch9_进阶_生产化.py           # 真起服务(gr.State 计数器 + 流式)
================================================================================
"""
import sys
import time
import gradio as gr


# ==============================================================================
# ① gr.State：会话状态(每个浏览器会话独立，不串号)
# ==============================================================================
def add_one(history):
    history = (history or 0) + 1                       # history 是本会话私有的 State
    return history, f"你点了 {history} 次（这个计数每个用户独立）"


def build_state_demo():
    with gr.Blocks(title="会话状态") as demo:
        gr.Markdown("## gr.State 会话计数器")
        state = gr.State(0)                            # ★每个会话一份，互不干扰
        out = gr.Textbox(label="结果")
        gr.Button("点我 +1").click(add_one, inputs=state, outputs=[state, out])
    return demo


# ==============================================================================
# ② 流式输出：yield 边算边吐(大模型生成的体验基础)
# ==============================================================================
def slow_echo(message):
    reply = f"你说的是：{message}"
    acc = ""
    for ch in reply:                                   # 逐字 yield → 前端打字机效果
        acc += ch
        time.sleep(0.01)
        yield acc


# ==============================================================================
# ③ gr.Progress：长任务进度条
# ==============================================================================
def long_task(n, progress=gr.Progress()):
    total = 0
    for i in progress.tqdm(range(n), desc="计算中"):     # progress.tqdm 自动驱动前端进度条
        total += i
        time.sleep(0.005)
    return f"1+2+...+{n-1} = {total}"


def build_progress_demo():
    return gr.Interface(fn=long_task, inputs=gr.Slider(10, 100, value=50, step=10),
                        outputs=gr.Textbox(label="结果"), title="进度条(gr.Progress)")


# ==============================================================================
# ④ mount_gradio_app：把 Gradio 挂到 FastAPI(业务API + UI 一体)
# ==============================================================================
def build_fastapi_app():
    from fastapi import FastAPI
    app = FastAPI(title="业务API + Gradio")

    @app.get("/api/health")                            # 你自己的业务 REST API
    def health():
        return {"status": "ok"}

    demo = gr.Interface(fn=lambda x: x[::-1], inputs="text", outputs="text",
                        title="字符串反转(挂在 /gradio)")
    app = gr.mount_gradio_app(app, demo, path="/gradio")   # ★Gradio UI 挂到 /gradio
    return app


def build_chat_streaming():
    # ChatInterface + 流式：真实聊天机器人的标准形态
    def chat(message, history):
        acc = ""
        for ch in f"收到：{message}":
            acc += ch; time.sleep(0.005); yield acc
    return gr.ChatInterface(fn=chat, title="流式聊天")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "state"
    if arg == "smoke":
        # 构建全部 UI + 直接调函数验证业务逻辑(不起服务)
        build_state_demo(); build_progress_demo(); build_chat_streaming()
        app = build_fastapi_app()
        assert add_one(2) == (3, "你点了 3 次（这个计数每个用户独立）")
        assert list(slow_echo("hi"))[-1] == "你说的是：hi"
        routes = [r.path for r in app.routes]
        assert "/api/health" in routes and any("gradio" in r for r in routes)
        print("✅ 构建自检通过：gr.State / 流式 yield / gr.Progress / mount_gradio_app / ChatInterface。")
        print("   FastAPI 路由:", [r for r in routes if r in ("/api/health",) or "gradio" in r][:4], "...")
        # print("   鉴权上线：demo.launch(auth=('user','pwd'), auth_message='请登录', "
              # "analytics_enabled=False, share=False)")
        # print("面试：Q gr.State 解决什么? Q 流式怎么做? Q 怎么把 Gradio 挂到已有 FastAPI? (见 ../../../Chapter 9/)")
    elif arg == "fastapi":
        import uvicorn
        uvicorn.run(build_fastapi_app(), host="127.0.0.1", port=8000)   # 访问 /gradio 和 /api/health
    else:
        build_state_demo().launch()
