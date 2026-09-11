"""
================================================================================
 Agent 实战 · 案例5 · 三大框架对比：smolagents / LangGraph / LlamaIndex（整合 Agent 课 Ch3）
================================================================================
 同一个“会查天气的助手”，用三种主流框架各搭一遍，看它们的抽象差异(Agent 课 Ch3 Gala 特工的做法)：
   · smolagents ：CodeAgent(tools, model) —— 最简，LLM 写代码调工具。
   · LangGraph  ：StateGraph 显式搭“节点+边”的状态机 —— 最可控，适合复杂多步/条件分支流程。
   · LlamaIndex ：AgentWorkflow.from_tools_or_functions —— 数据/RAG 生态强，工作流式。

 本机现实：三框架都已装，但 Agent 大脑都需 LLM(Token/服务)才能真 run。故：
   · smoke：验证三框架可导入 + 共享工具本地真跑 + 三个“组装函数”已定义(真实代码，需模型才 run)。
   · 组装/记忆差异见下表和各 build_* 函数(真实写法)。
 跑：python3 案例5_多框架对比_smolagents_LangGraph_LlamaIndex.py smoke
================================================================================
"""
import sys
import random


# 共享的纯逻辑工具(三框架复用同一份实现，本地可跑)
def get_weather(city: str) -> str:
    """返回某城市的(演示用)天气。

    Args:
        city: 城市名。
    """
    cond = random.Random(hash(city) & 0xffff).choice(
        [("晴", 25), ("多云", 20), ("小雨", 15)])
    return f"{city}：{cond[0]}，{cond[1]}°C"


# ==============================================================================
# ① smolagents：CodeAgent + @tool（最简）
# ==============================================================================
def build_smolagents():
    from smolagents import CodeAgent, InferenceClientModel, tool
    weather_tool = tool(get_weather)                       # 普通函数一行变工具
    return CodeAgent(tools=[weather_tool], model=InferenceClientModel(), max_steps=5)


# ==============================================================================
# ② LangGraph：显式状态机(节点=assistant/tools，边=条件路由)
# ==============================================================================
def build_langgraph():
    from typing import TypedDict, Annotated
    from langchain_core.messages import AnyMessage
    from langchain_core.tools import Tool as LcTool
    from langgraph.graph.message import add_messages
    from langgraph.graph import START, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

    tools = [LcTool(name="get_weather", func=get_weather, description="查询城市天气。")]
    chat = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-Coder-32B-Instruct"))
    chat_with_tools = chat.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def assistant(state):
        return {"messages": [chat_with_tools.invoke(state["messages"])]}

    b = StateGraph(State)
    b.add_node("assistant", assistant)
    b.add_node("tools", ToolNode(tools))
    b.add_edge(START, "assistant")
    b.add_conditional_edges("assistant", tools_condition)   # LLM 想调工具就去 tools 节点
    b.add_edge("tools", "assistant")
    return b.compile()


# ==============================================================================
# ③ LlamaIndex：AgentWorkflow（工作流式，RAG 生态强）
# ==============================================================================
def build_llamaindex():
    from llama_index.core.agent.workflow import AgentWorkflow
    from llama_index.core.tools import FunctionTool
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    weather_tool = FunctionTool.from_defaults(get_weather)
    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
    return AgentWorkflow.from_tools_or_functions([weather_tool], llm=llm)


COMPARE = [
    ("抽象",   "CodeAgent(tools,model)",  "StateGraph 节点+边",       "AgentWorkflow.from_tools"),
    ("调用方式", "LLM 写 Python 代码",       "LLM 出工具调用→ToolNode",   "工作流事件驱动"),
    ("记忆",   "run(reset=False)",        "接 messages / MemorySaver", "传 Context 对象"),
    ("最适合", "快速搭、代码类任务",         "复杂多步/条件分支、可控",     "数据/RAG 密集应用"),
]


def smoke():
    # 共享工具本地自检：get_weather('北京') -> 形如 "北京：晴，25°C"(按 city 哈希确定性选择)
    assert "°C" in get_weather("上海")

    # print("\n>>> 三框架可导入 + 组装函数已定义(真实代码，run 需 LLM/Token)：")
    for name, mod in [("smolagents", "smolagents"), ("LangGraph", "langgraph"),
                      ("LlamaIndex", "llama_index")]:
        import importlib.util
        ok = importlib.util.find_spec(mod) is not None
        # print(f"  {name:12} 可导入={ok}")
        assert ok
    for fn in (build_smolagents, build_langgraph, build_llamaindex):
        assert callable(fn)

    # print("\n>>> 三框架对比：")
    # print(f"   {'维度':<8}{'smolagents':<26}{'LangGraph':<26}{'LlamaIndex'}")
    for dim, a, b, c in COMPARE:
        pass
        # print(f"   {dim:<8}{a:<26}{b:<26}{c}")
    print("\n✅ 案例5 跑通：共享工具本地可跑 + 三框架组装写法(smolagents/LangGraph/LlamaIndex)对比清楚。")
    # print("面试：Q 三框架怎么选? A 快速/代码任务→smolagents;复杂可控流程→LangGraph;数据RAG重→LlamaIndex。")


if __name__ == "__main__":
    smoke()
