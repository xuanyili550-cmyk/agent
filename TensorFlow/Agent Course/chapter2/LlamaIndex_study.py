from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 从环境变量中检索 HF_TOKEN
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
    provider="auto"
)

response = llm.complete("Hello, how are you?")
print(response)
# I am good, how can I help you today?

# LlamaIndex 中QueryEngine 构建智能体 RAG 工作流程的关键组件
#使用组件创建 RAG 管道
# RAG流程包含五个关键阶段，而这些阶段又会成为您构建的大多数大型应用程序的组成部分。这五个阶段分别是：
#
# 加载：这指的是将数据从其存储位置（无论是文本文件、PDF、其他网站、数据库还是 API）导入到您的工作流程中。LlamaHub 提供数百种集成方案供您选择。
# 索引：指的是创建一种允许查询数据的数据结构。对于逻辑语言模型（LLM）而言，这几乎总是意味着创建向量嵌入，即数据含义的数值表示。索引还可以指代许多其他元数据策略，以便根据属性轻松准确地找到上下文相关的数据。
# 存储：数据建立索引后，您需要存储索引以及其他元数据，以避免重新建立索引。
# 查询：对于任何给定的索引策略，您可以使用 LLM 和 LlamaIndex 数据结构进行查询，包括子查询、多步骤查询和混合策略。
# 评估：任何流程的关键步骤之一就是检查其相对于其他策略的有效性，或者在进行更改时进行评估。评估提供客观的衡量标准，用于衡量您对查询的响应的准确性、可靠性和速度。

#加载和嵌入文档
# 数据加载到 LlamaIndex 中主要有三种方法：
#
# SimpleDirectoryReader：一个内置的加载器，用于加载本地目录中的各种文件类型。
# LlamaParse: LlamaParse 是 LlamaIndex 的官方 PDF 解析工具，以托管 API 的形式提供。
# LlamaHub：一个包含数百个数据加载库的注册表，可以从任何来源摄取数据。

from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="./data")
documents = reader.load_data()


from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

# 创建包含转换的管道
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ]
)

# .py 脚本不支持顶层 await（notebook 才行），用 asyncio.run 包起来
import asyncio
nodes = asyncio.run(pipeline.arun(documents=[Document.example()]))

#文档的存储和索引
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,
)

#从向量存储和嵌入中创建这个索引
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


#使用提示和 LLM 查询 VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)
query_engine.query("What is the meaning of life?")
# 生命的意义是42
# 反应处理
# 在底层，查询引擎不仅使用 LLM 来回答问题，还使用ResponseSynthesizer策略来处理响应。同样，这完全可以自定义，但有三种主要策略开箱即用，效果良好：
#
# refine：通过依次遍历每个检索到的文本块来创建和完善答案。这会为每个节点/检索到的文本块单独调用一次 LLM。
# compact（默认）：类似于精炼，但会预先连接数据块，从而减少 LLM 调用次数。
# tree_summarize：通过遍历检索到的每个文本块，并创建答案的树状结构，从而创建详细的答案。


#评估和可观测性
# FaithfulnessEvaluator：通过检查答案是否得到上下文支持来评估答案的可靠性。
# AnswerRelevancyEvaluator：通过检查答案是否与问题相关来评估答案的相关性。
# CorrectnessEvaluator：通过检查答案是否正确来评估答案的正确性
from llama_index.core.evaluation import FaithfulnessEvaluator

query_engine = query_engine
llm =llm

# 查询索引
evaluator = FaithfulnessEvaluator(llm=llm)
response = query_engine.query(
    "What battles took place in New York City in the American Revolution?"
)
eval_result = evaluator.evaluate_response(response=response)
print(eval_result.passing)

import llama_index
import os

PHOENIX_API_KEY = "<PHOENIX_API_KEY>"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
llama_index.core.set_global_handler(
    "arize_phoenix",
    endpoint="https://llamatrace.com/v1/traces"
)


#在 LlamaIndex 中使用工具
# LlamaIndex 中有四种主要类型的工具
# FunctionTool将任何 Python 函数转换为代理可以使用的工具。它会自动理解函数的工作原理。
# QueryEngineTool：一种允许代理使用查询引擎的工具。由于代理是基于查询引擎构建的，因此它们也可以将其他代理用作工具。
# Toolspecs社区创建的工具集，通常包括用于特定服务（如 Gmail）的工具。
# Utility Tools：用于帮助处理来自其他工具的大量数据的特殊工具。

#创建函数工具FunctionTool
from llama_index.core.tools import FunctionTool

def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    print(f"Getting weather for {location}")
    return f"The weather in {location} is sunny"

tool = FunctionTool.from_defaults(
    get_weather,
    name="my_weather_tool",
    description="Useful for getting the weather for a given location.",
)
tool.call("New York")

#创建查询引擎工具
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(llm=llm)
tool = QueryEngineTool.from_defaults(query_engine, name="some useful name", description="some useful description")

#创建工具规范
from llama_index.tools.google import GmailToolSpec
tool_spec=GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()

#更详细地了解这些工具，我们可以查看metadata每个工具的详情。
[(tool.metadata.name, tool.metadata.description) for tool in tool_spec_list]

#LlamaIndex 中的模型上下文协议 (MCP)
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# 我们假设 127.0.0.1:8000 上运行着一个 MCP 服务器，或者您可以使用 MCP 客户端连接到您自己的 MCP 服务器。mcp_client
mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
mcp_tool = McpToolSpec(client=mcp_client)

# 课程省略了 get_agent 的定义，这里补上；Context 需从 workflow 导入
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context


async def get_agent(tools: McpToolSpec):
    tools = await tools.to_tool_list_async()
    agent = FunctionAgent(
        name="Agent",
        description="An agent that can work with Our Database software.",
        tools=tools,
        llm=llm,
        system_prompt="You are a helpful assistant that has access to a database.",
    )
    return agent


# 这段需要 127.0.0.1:8000/sse 上有一个运行中的 MCP 服务器
async def _run_mcp_demo():
    agent = await get_agent(mcp_tool)          # 获取代理
    agent_context = Context(agent)             # 创建代理上下文
    return agent, agent_context


# .py 脚本不支持顶层 await；没有 MCP 服务器时优雅跳过而不是崩掉整个文件
try:
    agent, agent_context = asyncio.run(_run_mcp_demo())
except Exception as e:
    print(f"[MCP demo skipped] 需要先启动 MCP 服务器 (127.0.0.1:8000/sse): {e}")

#初始化代理
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
# 定义示例工具 -- 类型注解、函数名和文档字符串都包含在解析后的模式中！
def multiply(a:int ,b:int)->int:
    """将两个整数相乘并返回结果整数"""
    return a*b
# 初始化 llm
llm = HuggingFaceInferenceAPI(model_name= "Qwen/Qwen2.5-Coder-32B-Instruct" )
# 初始化代理
agent=AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm
)
#默认情况下，智能体是无状态的，但它们可以使用Context对象来记住过去的交互
# 无状态 stateless —— 应为 await 而非 async，且 .py 需放进 async 函数
from llama_index.core.workflow import Context


async def _run_stateless():
    return await agent.run("What is 2 times 2?")


# 记住 —— 用 Context 让 agent 记住上一轮对话
async def _run_stateful():
    ctx = Context(agent)
    await agent.run("My name is Bob.", ctx=ctx)
    return await agent.run("What was my name again?", ctx=ctx)


print(asyncio.run(_run_stateless()))
print(asyncio.run(_run_stateful()))
#使用 QueryEngineTools 创建 RAG 代理
from  llama_index.core.tools import QueryEngineTool
# 如 LlamaIndex 组件部分所示
query_engine=index.as_query_engine(llm=llm,similarity_top_k=3)
query_engine_tool=QueryEngineTool(
    query_engine=query_engine,
    name='name',
    description="a specific description",
    return_direct=False,
)
query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. "
)

#创建多智能体系统
#LlamaIndex 中的代理也可以直接用作其他代理的工具，以应对更复杂和自定义的场景
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)

# 定义一些工具
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


# 创建代理配置#注意：这里可以使用 FunctionAgent 或 ReActAgent。
# FunctionAgent 适用于具有函数调用 API 的 LLM。
# ReActAgent 适用于任何 LLM。
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use your tool to query a RAG system to answer information about XYZ",
    tools=[query_engine_tool],
    llm=llm
)

# 创建并运行工作流
agent = AgentWorkflow(
    agents=[calculator_agent, query_agent], root_agent="calculator"
)

# 运行系统
async def runs():
    return await agent.run(user_msg="Can you add 5 and 3?")
print(asyncio.run(runs()))

#在 LlamaIndex 中创建代理工作流
# 工作流程具有以下几个主要优势：
# 将代码清晰地组织成离散的步骤
# 事件驱动架构实现灵活的控制流
# 步骤之间的类型安全通信
# 内置状态管理
# 支持简单和复杂的代理交互
#todo 创建工作流程
from llama_index.core.workflow import StartEvent,StopEvent,Workflow,step
class MyWorkflow(Workflow):
    @step
    async def my_step(self,ev:StopEvent)-> StopEvent:
        # 在这里执行一些操作
        return StopEvent(result='Hello, world!')

w=MyWorkflow(timeout=10,verbose=False)
async def _runWork():
    return await w.run()
print(asyncio.run(_runWork()))
#连接多个步骤

from llama_index.core.workflow import Event

class ProcessingEvent(Event):
    intermediate_result: str

class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        # 处理初始数据
        return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # 使用中间结果
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)

w = MultiStepWorkflow(timeout=10, verbose=False)
async def _runWork():
    return await w.run()
print(asyncio.run(_runWork()))

#循环和分支
from llama_index.core.workflow import Event
import random
class ProcessingEvent(Event):
    intermediate_result: str

class LoopEvent(Event):
    loop_output: str

class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # 使用中间结果
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


w = MultiStepWorkflow(verbose=False)
async def result():
    return await w.run()
print(asyncio.run(result()))

#绘图工作流程
from llama_index.utils.workflow import draw_all_possible_flows

w = MultiStepWorkflow(verbose=False)
draw_all_possible_flows(w, "flow.html")

#国家管理
from llama_index.core.workflow import Context, StartEvent, StopEvent


@step
async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
    # 将查询结果存储在上下文中
    await ctx.store.set("query", "What is the capital of France?")

    # 根据上下文和事件执行某些操作
    val = ...

    # 从上下文中检索查询 query
    query = await ctx.store.get("query")

    return StopEvent(result=val)


#利用多代理工作流实现工作流程自动化
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# 定义一些工具
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# 我们可以直接传递函数，无需使用 FunctionTool——函数名/文档字符串会被解析以获取名称/描述。
multiply_agent = ReActAgent(
    name="multiply_agent",
    description="Is able to multiply two integers",
    system_prompt="A helpful assistant that can use a tool to multiply numbers.",
    tools=[multiply],
    llm=llm,
)

addition_agent = ReActAgent(
    name="add_agent",
    description="Is able to add two integers",
    system_prompt="A helpful assistant that can use a tool to add numbers.",
    tools=[add],
    llm=llm,
)

# 创建工作流程
workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent",
)

#response = await workflow.run(user_msg="Can you add 5 and 3?")

async def response():
    return await workflow.run(user_msg="Can you add 5 and 3?")
print(asyncio.run(response()))

from llama_index.core.workflow import Context

# 定义一些工具
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers."""
    # 更新计数
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)

    return a + b

async def multiply(ctx: Context, a: int, b: int) -> int:
    """Multiply two numbers."""
    # 更新计数
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)

    return a * b

...

workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent",
    initial_state={"num_fn_calls": 0},
    state_prompt="Current state: {state}. User message: {msg}",
)

# 运行工作流并附带上下文（.py 需放进 async 函数）
async def _run_stateful_workflow():
    ctx = Context(workflow)
    response = await workflow.run(user_msg="Can you add 5 and 3?", ctx=ctx)

    # 取出并查看状态
    state = await ctx.store.get("state")
    print(state["num_fn_calls"])
    return response


asyncio.run(_run_stateful_workflow())




















