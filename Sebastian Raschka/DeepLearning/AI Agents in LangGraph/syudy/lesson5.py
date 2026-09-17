
from dotenv import load_dotenv
_=load_dotenv()
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
MODEL = "qwen2.5"

from langgraph.graph import StateGraph,END
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import TypedDict,Annotated
from langchain_core.messages import AnyMessage,SystemMessage,HumanMessage,ToolMessage,AIMessage
import operator
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
_memory_cm=SqliteSaver.from_conn_string(":memory:")
memory=_memory_cm.__enter__()

from uuid import uuid4
def reduce_messages(left:list[AnyMessage],right:list[AnyMessage])->list[AnyMessage]:
    for message in right:
        if not message.id:
            message.id=str(uuid4())
    merged=left.copy()
    for message in right:
        for i ,existing in enumerate(merged):
            if existing.id==message.id:
                merged[i]=message
                break
        else:
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages:Annotated[list[AnyMessage],reduce_messages]

_search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)
tool=DuckDuckGoSearchResults(api_wrapper=_search_wrapper)
class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

prompt = """你是一个聪明的研究助手。请使用搜索引擎来查找信息。\
你可以多次调用搜索（可以一次并行发起多个，也可以分几轮依次调用）。\
只在你清楚自己要查什么的时候才去搜索。\
如果在追问之前需要先查一些信息，你也可以先查再问！请用中文回答。
"""
model = ChatOpenAI(model=MODEL, base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
abot = Agent(model, [tool], system=prompt, checkpointer=memory)
messages = [HumanMessage(content="旧金山现在天气怎么样？")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
abot.graph.get_state(thread)
abot.graph.get_state(thread).next
for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)
abot.graph.get_state(thread)
abot.graph.get_state(thread).next
messages = [HumanMessage("洛杉矶现在天气怎么样？")]
abot.graph.get_state(thread)
current_values = abot.graph.get_state(thread)
current_values.values['messages'][-1]
current_values.values['messages'][-1].tool_calls
_id = current_values.values['messages'][-1].tool_calls[0]['id']
current_values.values['messages'][-1].tool_calls = [
    {'name': 'duckduckgo_results_json',
  'args': {'query': '路易斯安那州现在的天气'},
  'id': _id}
]
abot.graph.update_state(thread, current_values.values)
abot.graph.get_state(thread)
for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)
states = []
for state in abot.graph.get_state_history(thread):
    print(state)
    states.append(state)
to_replay = states[-3]
for event in abot.graph.stream(None, to_replay.config):
    for k, v in event.items():
        print(v)
_id = to_replay.values['messages'][-1].tool_calls[0]['id']
to_replay.values['messages'][-1].tool_calls = [{'name': 'duckduckgo_results_json',
  'args': {'query': '洛杉矶现在的天气 accuweather'},
  'id': _id}]
branch_state = abot.graph.update_state(to_replay.config, to_replay.values)
for event in abot.graph.stream(None, branch_state):
    for k, v in event.items():
        if k != "__end__":
            print(v)
_id = to_replay.values['messages'][-1].tool_calls[0]['id']
state_update = {"messages": [ToolMessage(
    tool_call_id=_id,
    name="duckduckgo_results_json",
    content="洛杉矶现在 54 摄氏度",
)]}
branch_and_add = abot.graph.update_state(
    to_replay.config,
    state_update,
    as_node="action")
for event in abot.graph.stream(None, branch_and_add):
    for k, v in event.items():
        print(v)


class AgentState(TypedDict):
    lnode:str
    scratch: str
    count: Annotated[int, operator.add]
def node1(state: AgentState):
    print(f"node1, count:{state['count']}")
    return {"lnode": "node_1",
            "count": 1,
           }
def node2(state: AgentState):
    print(f"node2, count:{state['count']}")
    return {"lnode": "node_2",
            "count": 1,
           }
def should_continue(state):
    return state["count"] < 3
builder = StateGraph(AgentState)
builder.add_node("Node1", node1)
builder.add_node("Node2", node2)

builder.add_edge("Node1", "Node2")
builder.add_conditional_edges("Node2",
                              should_continue,
                              {True: "Node1", False: END})
builder.set_entry_point("Node1")
_memory_cm2 = SqliteSaver.from_conn_string(":memory:")
memory = _memory_cm2.__enter__()
graph = builder.compile(checkpointer=memory)
thread = {"configurable": {"thread_id": str(1)}}
graph.invoke({"count":0, "scratch":"hi"},thread)
graph.get_state(thread)
for state in graph.get_state_history(thread):
    print(state, "\n")
states = []
for state in graph.get_state_history(thread):
    states.append(state.config)
    print(state.config, state.values['count'])
states[-3]
graph.get_state(states[-3])
graph.invoke(None, states[-3])
thread = {"configurable": {"thread_id": str(1)}}
for state in graph.get_state_history(thread):
    print(state.config, state.values['count'])

thread2 = {"configurable": {"thread_id": str(2)}}
graph.invoke({"count":0, "scratch":"hi"},thread2)

print(graph.get_graph().draw_mermaid())

states2 = []
for state in graph.get_state_history(thread2):
    states2.append(state.config)
    print(state.config, state.values['count'])

save_state = graph.get_state(states2[-3])
save_state

save_state.values["count"] = -3
save_state.values["scratch"] = "hello"
save_state

graph.update_state(thread2,save_state.values)
for i, state in enumerate(graph.get_state_history(thread2)):
    if i >= 3:
        break
    print(state, '\n')

graph.update_state(thread2,save_state.values, as_node="Node1")
for i, state in enumerate(graph.get_state_history(thread2)):
    if i >= 3:
        break
    print(state, '\n')
graph.invoke(None,thread2)
for state in graph.get_state_history(thread2):
    print(state,"\n")


from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_tavily import TavilySearch
_memory_cm = SqliteSaver.from_conn_string(":memory:")
memory = _memory_cm.__enter__()

def reduce_messages(left:list[AnyMessage],right:list[AnyMessage])->list[AnyMessage]:
    for message in right:
        if not message.id:
            message.id=str(uuid4())
    merged=left.copy()
    for message in right:
        for i ,existing in enumerate(merged):
            if message.id==message.id:
                merged[i]=message
                break
        else:
            merged.append(message)
    return merged
class AgentState(TypedDict):
    messages:Annotated[list[AnyMessage],reduce_messages]

tool=TavilySearch(max_results=2)
class Agent:
    def __init__(self,model,tools,system="",checkpointer=None):
        self.system=system
        graph=StateGraph(AgentState)
        graph.add_node("llm",self.call_openai)
        graph.add_node('action',self.take_action)
        graph.add_conditional_edges('llm',self.exists_action,{True:'action',False:END})
        graph.add_edge("action",'llm')
        graph.set_entry_point("llm")
        self.graph=graph.compile(checkpointer=checkpointer,interrupt_before=["action"])
        self.tools={t.name:t for t in tools}
        self.model=model.bind_tools(tools)
    def call_openai(self,state:AgentState):
        messages=state['messages']
        if self.system:
            messages=[SystemMessage(content=self.system)]+messages
        message=self.model.invoke(messages)
        return {"messages":[message]}

    def exists_action(self, state: AgentState):
        result=state['messages'][-1]
        return len(result.tool_calls)>0
    def take_action(self, state: AgentState):
        tool_calls=state['messages'].tool_calls
        results=[]
        for t in tool_calls:
           result=self.tools[t['name']].invoke(t['args'])
           results.append(ToolMessage(tool_call_id=t['id'],name=t['name'],content=str[result]))
        return {"messages":results}

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(model="gpt-3.5-turbo")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)
messages = [HumanMessage(content="Whats the weather in SF?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)
messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
while abot.graph.get_state(thread).next:
    print("\n", abot.graph.get_state(thread),"\n")
    _input = input("proceed?")
    if _input != "y":
        print("aborting")
        break
    for event in abot.graph.stream(None, thread):
        for v in event.values():
            print(v)

_id = current_values.values['messages'][-1].tool_calls[0]['id']
current_values.values['messages'][-1].tool_calls = [
    {'name': 'tavily_search_results_json',
  'args': {'query': 'current weather in Louisiana'},
  'id': _id}
]
