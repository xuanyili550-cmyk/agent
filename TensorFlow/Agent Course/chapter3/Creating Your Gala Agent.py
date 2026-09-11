"""
组装 Alfred：完整的 Gala 特工 —— 同一个特工的三套框架实现。
每块：初始化 4 个工具（嘉宾检索 + 天气 + Hub统计 + 网络搜索）→ 组装 Alfred → 跑示例查询。

工具/检索器的定义放在配套模块里：
  - tools.py     ：天气 / Hub统计 / DuckDuckGo 搜索（按框架不同后缀导出）
  - retriever.py ：嘉宾 BM25 检索（惰性加载函数）

按块单独运行；运行任意一块都需要 .env 里的有效 HF_TOKEN。
"""

import os
from dotenv import load_dotenv

load_dotenv()  # 从 .env 读取 HF_TOKEN


# =====================================================================
# #smolagents
# =====================================================================
from smolagents import CodeAgent, InferenceClientModel
from tools import DuckDuckGoSearchTool, WeatherInfoTool, HubStatsTool
from retriever import load_guest_dataset

model = InferenceClientModel()

# 初始化 4 个工具（smolagents 版）
search_tool = DuckDuckGoSearchTool()
weather_info_tool = WeatherInfoTool()
hub_stats_tool = HubStatsTool()
guest_info_tool = load_guest_dataset()  # 加载嘉宾数据集并建检索工具

# 组装 Alfred；add_base_tools 加内置基础工具，planning_interval 每 3 步做一次规划
alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool],
    model=model,
    add_base_tools=True,
    planning_interval=3,
)

# 示例查询（CodeAgent.run 是同步的）
for query in [
    "Tell me about 'Lady Ada Lovelace'",                                                         # 查嘉宾
    "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?",  # 天气
    "One of our guests is from Qwen. What can you tell me about their most popular model?",      # Hub 统计
    "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. "
    "Can you help me prepare for this conversation?",                                            # 组合多工具
]:
    print("🎩 Alfred's Response:")
    print(alfred.run(query))
    print()

# --- 高级功能：对话记忆（smolagents 靠 reset=False 保留上一轮上下文） ---
alfred_with_memory = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool],
    model=model,
    add_base_tools=True,
    planning_interval=3,
)
print("🎩 Alfred's First Response:")
print(alfred_with_memory.run("Tell me about Lady Ada Lovelace."))
print("🎩 Alfred's Second Response:")
print(alfred_with_memory.run("What projects is she currently working on?", reset=False))


# =====================================================================
# #llam-index
# =====================================================================
import asyncio
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from tools import search_tool as li_search_tool, weather_info_tool_li, hub_stats_tool_li
from retriever import load_guest_info_tool_llama

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
guest_info_tool = load_guest_info_tool_llama()

alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool, li_search_tool, weather_info_tool_li, hub_stats_tool_li],
    llm=llm,
)


# AgentWorkflow.run 是异步的 → 用 asyncio.run 包起来；Context 让多轮对话共享记忆
async def _run_llama_demo():
    ctx = Context(alfred)
    print("🎩 Alfred's First Response:")
    print(await alfred.run("Tell me about Lady Ada Lovelace.", ctx=ctx))
    print("🎩 Alfred's Second Response:")
    print(await alfred.run("What projects is she currently working on?", ctx=ctx))


asyncio.run(_run_llama_demo())


# =====================================================================
# #langgraph
# =====================================================================
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from tools import DuckDuckGoSearchRun, weather_info_tool_lg, hub_stats_tool_lg
from retriever import load_guest_info_tool_langgraph

# token 从环境变量读
HUGGINGFACEHUB_API_TOKEN = os.getenv("HF_TOKEN")

# 初始化 4 个工具（langgraph 版）
search_tool = DuckDuckGoSearchRun()
guest_info_tool = load_guest_info_tool_langgraph()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [guest_info_tool, search_tool, weather_info_tool_lg, hub_stats_tool_lg]
chat_with_tools = chat.bind_tools(tools)


# 图状态：messages 用 add_messages 做归约，实现追加式对话
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState):
    return {"messages": [chat_with_tools.invoke(state["messages"])]}


# 搭图：assistant 决定是否调工具，tools_condition 负责路由
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

# 第一轮
response = alfred.invoke({
    "messages": [HumanMessage(
        content="Tell me about 'Lady Ada Lovelace'. What's her background and how is she related to me?"
    )]
})
print("🎩 Alfred's Response:")
print(response["messages"][-1].content)
print()

# 第二轮：把上一轮完整 messages 接上再追问，实现多轮记忆
response = alfred.invoke({
    "messages": response["messages"] + [HumanMessage(content="What projects is she currently working on?")]
})
print("🎩 Alfred's Response:")
print(response["messages"][-1].content)


# =====================================================================
# 三框架的"记忆"设计差异（都没把内存直接耦合进 agent）：
#   - smolagents ：默认不跨 run 保留，需显式 reset=False
#   - llama-index：运行时显式传入 Context 对象管理记忆
#   - langgraph  ：可自己接上 messages，或用 MemorySaver 组件
# =====================================================================
