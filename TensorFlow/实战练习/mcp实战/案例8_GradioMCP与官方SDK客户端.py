"""
================================================================================
 MCP 实战 · 案例8 · Gradio 一键起 MCP 服务器 + 官方 SDK/Agent 客户端写法
================================================================================
 前面案例都是“手写 stdio 服务器 + subprocess 客户端”；本文件补上两块生态写法：
   ① Gradio MCP 服务器：给普通函数写好 docstring/类型注解，launch(mcp_server=True) 一行
      就把它暴露成 MCP 工具（Gradio 自动生成 schema + 起 SSE 端点 /gradio_api/mcp/sse）。
   ② 官方 SDK / Agent 客户端：真实项目里 Host 端不手写 subprocess，而是用
      smolagents.MCPClient 或 huggingface_hub 的微型 Agent 连接 MCP 服务器、自动选工具调用。
 说明：Gradio 工具函数本机可跑并自检；Agent 客户端需联网+模型/额外依赖，作【参考写法】给出。
 跑：python3 案例8_GradioMCP与官方SDK客户端.py smoke   # 构建 Gradio MCP demo + 工具函数自检
    python3 案例8_GradioMCP与官方SDK客户端.py          # 真起 Gradio MCP 服务器(需 gradio[mcp])
================================================================================
"""
import sys
import gradio as gr


# ==============================================================================
# ① 工具函数：docstring + 类型注解 = MCP 工具的 name/description/inputSchema
# ==============================================================================
def letter_counter(word: str, letter: str) -> int:
    """统计一个单词里某个字母出现的次数。

    Args:
        word: 要检查的单词或句子
        letter: 要统计的单个字母
    Returns:
        该字母出现的次数
    """
    return word.lower().count(letter.lower())


def sentiment_hint(text: str) -> str:
    """给一段文本一个极简的情绪提示(正面/负面/中性)。

    Args:
        text: 要判断的文本
    """
    pos = sum(w in text for w in ["good", "great", "love", "喜欢", "很好"])
    neg = sum(w in text for w in ["bad", "hate", "terrible", "差", "讨厌"])
    return "正面" if pos > neg else ("负面" if neg > pos else "中性")


def build_gradio_mcp():
    """构建一个 Gradio 应用；launch(mcp_server=True) 时这些函数会自动变成 MCP 工具。"""
    return gr.Interface(
        fn=letter_counter,
        inputs=[gr.Textbox(label="word"), gr.Textbox(label="letter")],
        outputs=gr.Number(label="count"),
        title="Letter Counter (Gradio MCP)",
        description="演示：普通函数 → launch(mcp_server=True) → 自动成为 MCP 工具。",
    )


# ==============================================================================
# ② 官方 SDK / Agent 客户端（真实代码，惰性导入；需联网+模型/额外依赖，本机不跑）
# ==============================================================================
def run_smolagents_sse(sse_url="http://127.0.0.1:7860/gradio_api/mcp/sse"):
    """用 smolagents 连 SSE 传输的 MCP 服务器(如上面的 Gradio MCP)，Agent 自动发现并调用工具。
    需 pip install 'smolagents[mcp]' + 可用的推理模型(InferenceClient/本地)。"""
    from smolagents import CodeAgent, InferenceClientModel      # 惰性导入
    from smolagents.mcp_client import MCPClient
    with MCPClient({"url": sse_url, "transport": "sse"}) as tools:
        agent = CodeAgent(tools=tools, model=InferenceClientModel())
        return agent.run("How many 'r' are in 'strawberry'?")   # 自动发现 letter_counter 并调用


def run_smolagents_stdio():
    """用 smolagents 连 stdio 传输的本地 MCP 服务器(把服务器当子进程拉起)。"""
    from smolagents import CodeAgent, InferenceClientModel, ToolCollection
    from mcp import StdioServerParameters
    server = StdioServerParameters(command="python3",
                                   args=["案例1_MCP工具服务器_NLP能力.py", "--server"])
    with ToolCollection.from_mcp(server, trust_remote_code=True) as tc:
        agent = CodeAgent(tools=[*tc.tools], model=InferenceClientModel())
        return agent.run("分析这句话的情感：This update is fantastic!")


async def run_hf_agent():
    """用 huggingface_hub 的微型 Agent 连 MCP 服务器(stdio)。需 pip install 'huggingface_hub[mcp]'。"""
    from huggingface_hub import Agent                           # 惰性导入
    agent = Agent(
        model="Qwen/Qwen2.5-72B-Instruct", provider="nebius",
        servers=[{"type": "stdio", "command": "python3",
                  "args": ["案例1_MCP工具服务器_NLP能力.py", "--server"]}],
    )
    await agent.load_tools()                     # 连接服务器、tools/list 发现工具
    chunks = []
    async for chunk in agent.run("统计 strawberry 里有几个 r"):
        chunks.append(chunk)                     # 模型 function calling → 调 MCP 工具 → 综合回答
    return chunks

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if arg == "smoke":
        build_gradio_mcp()                          # 能构建
        assert letter_counter("strawberry", "r") == 3
        assert sentiment_hint("this is great and I love it") == "正面"
        print("✅ Gradio MCP 工具函数自检通过(letter_counter/sentiment_hint)，Interface 构建成功。")
        # print("   真起 MCP 服务器：python3 案例8_GradioMCP与官方SDK客户端.py")
        # print("   (需 pip install 'gradio[mcp]'；起后 SSE 端点在 http://127.0.0.1:7860/gradio_api/mcp/sse)")
        # 官方 SDK 客户端是真实函数(需联网+模型，本机不调用，只确认已定义)：
        import inspect
        for fn in (run_smolagents_sse, run_smolagents_stdio, run_hf_agent):
            assert inspect.isfunction(fn) or inspect.iscoroutinefunction(fn)
        # print("\n—— 官方 SDK/Agent 客户端(真实代码，需联网+模型，本机不跑) ——")
        # print("  run_smolagents_sse()   连 SSE 服务器(smolagents.MCPClient + CodeAgent)")
        # print("  run_smolagents_stdio() 连 stdio 服务器(ToolCollection.from_mcp)")
        # print("  run_hf_agent()         huggingface_hub 微型 Agent(async, load_tools+run)")
        # print("面试：Q Gradio 怎么一键成 MCP 服务器? Q 生产里 Host 端怎么连 MCP(不手写subprocess)?"
              # " Q smolagents/hf Agent 连 MCP 的两种传输? (见 README_mcp实战导航.py)")
    else:
        # 真起 Gradio MCP 服务器：函数自动暴露成 MCP 工具(需 gradio[mcp])
        demo = gr.TabbedInterface(
            [build_gradio_mcp(),
             gr.Interface(fn=sentiment_hint, inputs="text", outputs="text", title="Sentiment Hint")],
            ["Letter Counter", "Sentiment"])
        demo.launch(mcp_server=True)
