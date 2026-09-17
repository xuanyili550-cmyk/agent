from dotenv import load_dotenv
_=load_dotenv()
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
MODEL = "qwen2.5"
from langgraph.graph import StateGraph,END
import operator
from typing import TypedDict,Annotated,List
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
_memory_cm=SqliteSaver.from_conn_string(":memory:")
memory=_memory_cm.__enter__()

class AgentState(TypedDict):
    task:str
    plan:str
    draft:str
    critique:str
    content:List[str]
    revision_number:int
    max_revisions:int

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model=MODEL, base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY, temperature=0)
PLAN_PROMPT = """你是一位资深写作专家，任务是为一篇文章撰写高层次的大纲。\
请针对用户提供的主题写出这样一份大纲。给出文章的大纲，并为各个部分附上相关的备注或写作说明。请用中文。"""
WRITER_PROMPT = """你是一位写作助手，任务是写出优秀的五段式文章。\
请针对用户的需求和最初的大纲，写出尽可能好的文章。\
如果用户提供了批改意见，请据此给出修改后的新版本。\
按需充分利用下面提供的参考资料，并用中文写作：

------

{content}"""
REFLECTION_PROMPT = """你是一位批改作文的老师。\
请针对用户提交的文章给出点评和改进建议。\
提供详细的建议，包括对篇幅、深度、文风等方面的要求。请用中文。"""
RESEARCH_PLAN_PROMPT = """你是一名研究员，负责为撰写下面这篇文章提供可用的信息。\
请生成一组搜索关键词，用来收集任何相关信息。最多只生成 3 条关键词。"""
RESEARCH_CRITIQUE_PROMPT = """你是一名研究员，负责为完成下面提出的修改要求提供可用的信息。\
请生成一组搜索关键词，用来收集任何相关信息。最多只生成 3 条关键词。"""

from pydantic import BaseModel
class Queries(BaseModel):
    queries:List[str]

from duckduckgo_search import DDGS

class DuckDuckGoResearchClient:
    def __init__(self):
        self._ddg = DDGS()
    def search(self, query, max_results=2):
        try:
            hits=self._ddg.text(query,max_results=max_results) or []
        except Exception as e:
            hits = []
        return {"results": [{"content": h.get("body", "")} for h in hits]}
tavily=DuckDuckGoResearchClient()

def plan_node(state:AgentState):
    messages=[
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response=model.invoke(messages)
    return {"plan":response.content}
def research_plan_node(state: AgentState):
    queries=model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content=state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}
def generation_node(state:AgentState):
    content="\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\n这是我的写作大纲：\n\n{state['plan']}")
    message=[
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message
    ]
    response=model.invoke(message)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }
def reflection_node(state: AgentState):
    messages=[
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response=model.invoke(messages)
    return {"critique",response.content}
def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}
def should_continue(state:AgentState):
    if state['revision_number']>state['max_revisions']:
        return END
    return 'reflect'
builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)
builder.set_entry_point("planner")
builder.add_conditional_edges(
    "generate",
    should_continue,
    {END: END, "reflect": "reflect"}
)
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")

builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")
graph = builder.compile(checkpointer=memory)
print(graph.get_graph().draw_mermaid())
thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream({
    'task': "langchain 和 langsmith 有什么区别？",
    "max_revisions": 2,
    "revision_number": 1,
}, thread):
    print(s)
import warnings
warnings.filterwarnings("ignore")
from helper import ewriter, writer_gui
MultiAgent = ewriter()
app = writer_gui(MultiAgent.graph)
app.launch()