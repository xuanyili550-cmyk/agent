"""
智能检索增强生成（Agentic RAG）—— 同一个任务的三套框架实现
任务三步：① 加载并准备嘉宾数据集 → ② 创建 BM25 检索工具 → ③ 把工具接入 Alfred 智能体

下面按框架分成三大块：#smolagents / #llama-index / #langgraph。
三块各自独立、可单独运行（在 IDE 里选中某一块 Run 即可）；
注意三块会重复定义 docs / Document / Tool / BM25Retriever 等同名变量，这是"三框架对照"的固有现象，
按块单独跑不受影响。运行任意一块都需要 .env 里的有效 HF_TOKEN。
"""

import os
from dotenv import load_dotenv

load_dotenv()  # 从 .env 读取 HF_TOKEN


# =====================================================================
# #smolagents
# =====================================================================
import datasets
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool, CodeAgent, InferenceClientModel

# --- ① 加载数据集，并把每条嘉宾记录转成 langchain 的 Document ---
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}",
        ]),
        metadata={"name": guest["name"]},
    )
    for guest in guest_dataset
]


# --- ② 创建检索工具：继承 smolagents.Tool，内部用 BM25 检索 ---
class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about.",
        }
    }
    output_type = "string"

    def __init__(self, docs):
        self.is_initialized = False
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str):
        results = self.retriever.invoke(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        else:
            return "No matching guest information found."


guest_info_tool = GuestInfoRetrieverTool(docs)

# --- ③ 接入 Alfred（smolagents 的 CodeAgent，run 是同步的） ---
model = InferenceClientModel()
alfred = CodeAgent(tools=[guest_info_tool], model=model)

response = alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")
print("🎩 Alfred's Response:")
print(response)


# =====================================================================
# #llam-index
# =====================================================================
import asyncio
import datasets
from llama_index.core.schema import Document
from llama_index.core.tools import FunctionTool
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# --- ① 加载数据集，转成 llama_index 的 Document（这里按列取值 guest_dataset['col'][i]） ---
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")
docs = [
    Document(
        text="\n".join([
            f"Name: {guest_dataset['name'][i]}",
            f"Relation: {guest_dataset['relation'][i]}",
            f"Description: {guest_dataset['description'][i]}",
            f"Email: {guest_dataset['email'][i]}",
        ]),
        metadata={"name": guest_dataset['name'][i]},
    )
    for i in range(len(guest_dataset))
]

# --- ② 创建检索工具：普通函数 + FunctionTool 包装，检索器用 llama_index 的 BM25 ---
bm25_retriever = BM25Retriever.from_defaults(nodes=docs)


def get_guest_info_retriever(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.retrieve(query)
    if results:
        return "\n\n".join([doc.text for doc in results[:3]])
    else:
        return "No matching guest information found."


guest_info_tool = FunctionTool.from_defaults(get_guest_info_retriever)

# --- ③ 接入 Alfred（AgentWorkflow，run 是异步的 → .py 里用 asyncio.run 包起来） ---
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool],
    llm=llm,
)


async def _run_llama_alfred():
    return await alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")


response = asyncio.run(_run_llama_alfred())
print("🎩 Alfred's Response:")
print(response)


# =====================================================================
# #langgraph
# =====================================================================
import datasets
from typing import TypedDict, Annotated
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.tools import Tool
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# --- ① 加载数据集，转成 langchain 的 Document ---
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}",
        ]),
        metadata={"name": guest["name"]},
    )
    for guest in guest_dataset
]

# --- ② 创建检索工具：普通函数 + langchain 的 Tool 包装 ---
bm25_retriever = BM25Retriever.from_documents(docs)


def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."


guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation.",
)

# --- ③ 接入 Alfred（用 langgraph 手搭 assistant⇄tools 的状态图） ---
# token 从环境变量读，别硬编码（原代码里 HUGGINGFACEHUB_API_TOKEN 是未定义变量 → 爆红）
HUGGINGFACEHUB_API_TOKEN = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [guest_info_tool]
chat_with_tools = chat.bind_tools(tools)


# 定义图的状态：messages 用 add_messages 做归约，实现追加式对话
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }


# 搭图：assistant 节点决定是否调工具，tools_condition 负责路由
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # 最新消息需要工具就路由到 tools，否则直接回答
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
response = alfred.invoke({"messages": messages})

print("🎩 Alfred's Response:")
print(response["messages"][-1].content)


# =====================================================================
# 后续可改进方向（三框架通用）：
#   - 检索器换成更强的算法，例如 sentence-transformers 语义检索
#   - 加入对话记忆，让 Alfred 记住之前的互动
#   - 结合网络搜索，获取陌生客人的最新信息
#   - 整合多个索引，从可靠来源拿更完整的信息
# =====================================================================

###############################################################################
# 第二节：为代理构建并集成多个工具（DuckDuckGo 搜索 + 假天气 + HF Hub 下载统计）
# 同样按三框架分块，每块：造 3 个工具 → 一起接入 Alfred
###############################################################################

# =====================================================================
# #smolagents
# =====================================================================
from smolagents import DuckDuckGoSearchTool

# --- 工具1：DuckDuckGo 搜索（smolagents 内置，直接调用即可） ---
search_tool = DuckDuckGoSearchTool()
results = search_tool("Who's the current President of France?")  # 用法示例
print(results)

# --- 工具2：自定义天气工具（假数据，演示如何继承 Tool 自造工具） ---
from smolagents import Tool
import random


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches dummy weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for."
        }
    }
    output_type = "string"

    def forward(self, location: str):
        # 假的天气数据
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20}
        ]
        # 随机选一种天气
        data = random.choice(weather_conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# 初始化工具
weather_info_tool = WeatherInfoTool()

# --- 工具3：HF Hub 下载统计（查某作者下载量最高的模型） ---
from smolagents import Tool
from huggingface_hub import list_models


class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from."
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
            # 列出该作者的模型，按下载量排序
            models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

            if models:
                model = models[0]
                return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
            else:
                return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"


# 初始化工具
hub_stats_tool = HubStatsTool()

# 示例用法
print(hub_stats_tool("facebook"))  # 示例：查 facebook 下载量最高的模型

# --- 集成：三个工具一起交给 Alfred（CodeAgent.run 同步） ---
from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel()
alfred = CodeAgent(
    tools=[search_tool, weather_info_tool, hub_stats_tool],
    model=model
)
response = alfred.run("What is Facebook and what's their most popular model?")

print("🎩 Alfred's Response:")
print(response)

# =====================================================================
# #llam-index
# =====================================================================
import asyncio
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.tools import FunctionTool

# --- 工具1：DuckDuckGo 搜索（llama-index 的 ToolSpec 包一层成 FunctionTool） ---
tool_spec = DuckDuckGoSearchToolSpec()
search_tool = FunctionTool.from_defaults(tool_spec.duckduckgo_full_search)
response = search_tool("Who's the current President of France?")  # 示例用法
print(response.raw_output[-1]['body'])

# --- 工具2：自定义天气工具（普通函数 + FunctionTool 包装） ---
import random
from llama_index.core.tools import FunctionTool


def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # 假的天气数据
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # 随机选一种天气
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# 初始化工具
weather_info_tool = FunctionTool.from_defaults(get_weather_info)

# --- 工具3：HF Hub 下载统计 ---
from llama_index.core.tools import FunctionTool
from huggingface_hub import list_models


def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # 列出该作者的模型，按下载量排序
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

# 初始化工具
hub_stats_tool = FunctionTool.from_defaults(get_hub_stats)
print(hub_stats_tool("facebook"))  # 示例：查 facebook 下载量最高的模型

# --- 集成：三个工具一起接入 Alfred（AgentWorkflow.run 是异步的 → 用 asyncio.run 包起来） ---
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
alfred = AgentWorkflow.from_tools_or_functions(
    [search_tool, weather_info_tool, hub_stats_tool],
    llm=llm
)


async def _run_llama_alfred_tools():
    return await alfred.run("What is Facebook and what's their most popular model?")


response = asyncio.run(_run_llama_alfred_tools())
print("🎩 Alfred's Response:")
print(response)
# =====================================================================
# #langgraph
# =====================================================================
import os
from langchain_community.tools import DuckDuckGoSearchRun

# --- 工具1：DuckDuckGo 搜索（langchain_community 内置，用 .invoke 调用） ---
search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke("Who's the current President of France?")
print(results)

# --- 工具2：自定义天气工具（普通函数 + langchain 的 Tool 包装） ---
from langchain_core.tools import Tool
import random


def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # 假的天气数据
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # 随机选一种天气
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# 初始化工具
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)

# --- 工具3：HF Hub 下载统计 ---
from langchain_core.tools import Tool
from huggingface_hub import list_models


def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # 列出该作者的模型，按下载量排序
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

# 初始化工具
hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."
)

print(hub_stats_tool.invoke("facebook"))  # 示例：查 facebook 下载量最高的模型

# --- 集成：用 langgraph 手搭 assistant⇄tools 状态图，把三个工具接入 Alfred ---
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# token 从环境变量读（这块单独跑时也能拿到）
HUGGINGFACEHUB_API_TOKEN = os.getenv("HF_TOKEN")

# 生成带工具的 chat 接口
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [search_tool, weather_info_tool, hub_stats_tool]
chat_with_tools = chat.bind_tools(tools)

# 定义图状态 AgentState 与整张 Agent 图
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

## 状态图
builder = StateGraph(AgentState)

# 定义节点：真正干活的地方
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# 定义边：决定控制流如何流转
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # 若最新消息需要工具，就路由到 tools
    # 否则直接回答
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

messages = [HumanMessage(content="Who is Facebook and what's their most popular model?")]
response = alfred.invoke({"messages": messages})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)