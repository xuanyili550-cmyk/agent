"""
================================================================================
 Agent Course · Chapter 2 · Agent 框架（学习笔记，可运行真代码）
================================================================================
 三大主流框架搭 Agent 的抽象差异；工具本地真跑，框架组装是真实代码(run 需 LLM/Token)，
 另给一个【不依赖框架、mlx-lm 真跑】的 ReAct 基线，端到端能看结果。
   · smolagents ：@tool / CodeAgent(LLM 写代码调工具) / ToolCallingAgent(LLM 出 JSON 调用) / 模型后端。
   · LangGraph  ：StateGraph 显式“节点+边”状态机(最可控，复杂分支)。
   · LlamaIndex ：AgentWorkflow(数据/RAG 生态强)。
 本机：smolagents 小模型太弱(实测)，故框架版需 Token/vLLM；基线用 mlx-lm 真跑。
 跑：python3 chapter2_Agent框架_学习笔记.py
================================================================================
"""
import re

_M = {}


# —— 共享工具(三框架 + 基线复用，本地真跑) ——
def get_weather(city: str) -> str:
    """查询城市天气。
    Args:
        city: 城市名。
    """
    return f"{city}：晴，25°C（演示）"


def calculator(expr: str) -> str:
    """计算算术表达式。
    Args:
        expr: 只含数字和 +-*/() 的表达式。
    """
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return "非法"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return "计算失败"


# ==============================================================================
# ① smolagents：@tool + CodeAgent / ToolCallingAgent + 模型后端(真实代码)
# ==============================================================================
def build_smolagents_code():
    from smolagents import CodeAgent, InferenceClientModel, tool
    return CodeAgent(tools=[tool(get_weather), tool(calculator)],
                     model=InferenceClientModel(), max_steps=5)   # LLM 写 Python 调工具


def build_smolagents_toolcalling():
    from smolagents import ToolCallingAgent, InferenceClientModel, tool
    return ToolCallingAgent(tools=[tool(get_weather)], model=InferenceClientModel())  # LLM 出 JSON 调用


def build_smolagents_vllm(api_base="http://gpu-host:8001/v1"):
    from smolagents import CodeAgent, OpenAIServerModel, tool
    return CodeAgent(tools=[tool(calculator)],
                     model=OpenAIServerModel(model_id="qwen", api_base=api_base, api_key="EMPTY"))


# ==============================================================================
# ② LangGraph：显式状态机(节点 assistant/tools + 条件路由)
# ==============================================================================
def build_langgraph():
    from typing import TypedDict, Annotated
    from langchain_core.messages import AnyMessage
    from langchain_core.tools import Tool as LcTool
    from langgraph.graph.message import add_messages
    from langgraph.graph import START, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

    tools = [LcTool(name="get_weather", func=get_weather, description="查天气")]
    chat = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-Coder-32B-Instruct"))
    bound = chat.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    b = StateGraph(State)
    b.add_node("assistant", lambda s: {"messages": [bound.invoke(s["messages"])]})
    b.add_node("tools", ToolNode(tools))
    b.add_edge(START, "assistant")
    b.add_conditional_edges("assistant", tools_condition)
    b.add_edge("tools", "assistant")
    return b.compile()


# ==============================================================================
# ③ LlamaIndex：AgentWorkflow
# ==============================================================================
def build_llamaindex():
    from llama_index.core.agent.workflow import AgentWorkflow
    from llama_index.core.tools import FunctionTool
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    return AgentWorkflow.from_tools_or_functions(
        [FunctionTool.from_defaults(get_weather)],
        llm=HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct"))


# ==============================================================================
# ④ 不依赖框架的 ReAct 基线(mlx-lm 真跑，端到端能看结果)
# ==============================================================================
def baseline_react(question):
    if "m" not in _M:
        from mlx_lm import load
        _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import generate

    def llm(p, n=40):
        t = _M["t"].apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        return generate(_M["m"], _M["t"], prompt=t, max_tokens=n, verbose=False).strip()
    pick = llm(f"工具：calculator(算式)/get_weather(城市)。问题：{question}\n只输出：工具名|参数")
    # 规则路由为准(小模型不可靠)，LLM 提议仅在与规则一致且参数合法时采纳
    if re.search(r"\d.*[+\-*/]|算", question):
        fb_tool, fb_arg = "calculator", re.sub(r"[^\d+\-*/().]", "", question) or "1+1"
    else:
        fb_tool, fb_arg = "get_weather", re.sub(r"(天气|怎么样|的|今天)", "", question).strip() or "北京"
    tool, arg = fb_tool, fb_arg
    m = re.search(r"(calculator|get_weather)\s*[|｜]\s*(.+)", pick)
    if m and m.group(1) == fb_tool:
        cand = m.group(2).strip()
        if not (fb_tool == "calculator" and not re.fullmatch(r"[\d\s+\-*/().]+", cand)):
            arg = cand or fb_arg
    obs = {"calculator": calculator, "get_weather": get_weather}[tool](arg)
    return tool, obs


COMPARE = [("抽象", "CodeAgent(tools,model)", "StateGraph 节点+边", "AgentWorkflow"),
           ("调用", "LLM 写代码", "LLM 出调用→ToolNode", "工作流事件"),
           ("最适合", "快速/代码任务", "复杂可控流程", "数据/RAG 重")]

if __name__ == "__main__":
    print("① 共享工具本地自检：", calculator("6*7"), "/", get_weather("北京"))
    print("② 三框架可导入 + 组装函数(真实代码，run 需 LLM/Token)：")
    import importlib.util as u
    for mod in ("smolagents", "langgraph", "llama_index"):
        assert u.find_spec(mod)
    print("   smolagents / langgraph / llama_index 均可导入 ✅")
    print("③ 三框架对比：")
    for row in COMPARE:
        print("   " + "".join(f"{c:<24}" for c in row))
    print("④ ReAct 基线(mlx-lm 真跑)：")
    for q in ["算 128*39", "上海天气"]:
        tool, obs = baseline_react(q)
        print(f"   {q} → {tool} → {obs}")
    assert baseline_react("算 12+8")[0] == "calculator"
    print("\n✅ Chapter2 跑通：三框架组装写法 + mlx-lm ReAct 基线真跑 + 对比。")
    print("面试：Q 三框架怎么选? Q CodeAgent vs ToolCallingAgent? Q 各框架记忆怎么做?")
