import asyncio

from dotenv import load_dotenv
_=load_dotenv()
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
MODEL = "qwen2.5"
from langgraph.graph import StateGraph,END
from typing import TypedDict,Annotated
import operator
from langchain_core.messages import AnyMessage,SystemMessage,ToolMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

_seach_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3)
tool=DuckDuckGoSearchResults(api_wrapper=_seach_wrapper)

class AgentState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]

from langgraph.checkpoint.sqlite import SqliteSaver

_memory_cm=SqliteSaver.from_conn_string(":memory:")
memory=_memory_cm.__enter__()


class Agent:
    def __init__(self,model,tools,checkpointer,system=""):
        self.system=system
        graph=StateGraph(AgentState)
        graph.add_node("llm",self.call_openai)
        graph.add_node("action",self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True:"action",False:END}
        )
        graph.add_edge("action","llm")
        graph.set_entry_point("llm")
        self.graph=graph.compile(checkpointer=checkpointer)
        self.tools={t.name:t for t in tools}
        self.model=model.bind_tools(tools)

    def call_openai(self,state:AgentState):
        messages=state["messages"]
        if self.system:
            messages=[SystemMessage(content=self.system)]+messages
        message=self.model.invoke(messages)
        return {"messages":[message]}
    def exists_action(self,state:AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls)>0
    def take_action(self,state:AgentState):
        tool_calls=state["messages"][-1].tool_calls
        results=[]
        for t in tool_calls:
            result=self.tools[t["name"]].invoke(t["args"])
            results.append(ToolMessage(tool_call_id=t['id'],name=t['name'],content=str(result)))
        return {"messages":results}


prompt = """你是一个聪明的研究助手。请使用搜索引擎来查找信息。\
你可以多次调用搜索（可以一次并行发起多个，也可以分几轮依次调用）。\
只在你清楚自己要查什么的时候才去搜索。\
如果在追问之前需要先查一些信息，你也可以先查再问！请用中文回答。
"""

model=ChatOpenAI(model=MODEL,base_url=OLLAMA_BASE_URL,api_key=OLLAMA_API_KEY,temperature=0)
abot=Agent(model,[tool],checkpointer=memory,system=prompt)
messages=[HumanMessage(content="旧金山现在天气怎么样？")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages":messages},thread):
    for v in event:
        print(event)

messages = [HumanMessage(content="哪个更暖和？")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

messages = [HumanMessage(content="哪个更暖和？")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)


from  langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
_async_memory_cm=AsyncSqliteSaver.from_conn_string(":memory:")
async def memory():
        return await _async_memory_cm.__aenter__()
abot=Agent(model,[tool],checkpointer=memory,system=prompt)
messages = [HumanMessage(content="旧金山现在天气怎么样？")]
thread = {"configurable": {"thread_id": "4"}}
async def astream_events():
   async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="|")

asyncio.run(astream_events())


#第二种
tool=TavilySearchResults(max_results=2)
class AgentState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]

_memory_cm=SqliteSaver.from_conn_string(":memory:")
memory=_memory_cm.__enter__()

class Agent:
    def __init__(self,model,tools,checkpointer,system=""):
        self.system=system
        graph=StateGraph(AgentState)
        graph.add_node("llm",self.call_openai)
        graph.add_node("action",self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True:"action",False:END}
        )
        graph.add_edge("action","llm")
        graph.set_entry_point("llm")
        self.graph=graph.compile(checkpointer=checkpointer)
        self.tools={t.name:t for t in tools}
        self.model=model.bind_tools(tools)

    def call_openai(self,state:AgentState):
        messages=state["messages"]
        if self.system:
            messages=[SystemMessage(content=self.system)]+messages
        message=self.model.invoke(messages)
        return {"messages":[message]}
    def exists_action(self,state:AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls)>0
    def take_action(self,state:AgentState):
        tool_calls=state['messages'][-1].tool_calls
        results=[]
        for t in tool_calls:
            result=self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'],name=t['name'],content=str(result)))
        return {"messages":results}

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

model=ChatOpenAI(model="gpt-4o")
abot=Agent(model,[tool],checkpointer=memory,system=prompt)
messages = [HumanMessage(content="What is the weather in sf?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v['messages'])


messages = [HumanMessage(content="What about in la?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

_async_memory_cm = AsyncSqliteSaver.from_conn_string(":memory:")
async def memory():
    return await _async_memory_cm.__aenter__()
abot = Agent(model, [tool], system=prompt, checkpointer=memory)
messages = [HumanMessage(content="What is the weather in SF?")]
thread = {"configurable": {"thread_id": "4"}}
async  def astream_events():
    async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="|")

asyncio.run(astream_events())