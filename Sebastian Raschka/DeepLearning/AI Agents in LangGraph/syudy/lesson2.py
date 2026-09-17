from dotenv import load_dotenv
_=load_dotenv()
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
MODEL = "qwen2.5"

from langgraph.graph import StateGraph,END
from typing import TypedDict,Annotated
import operator
from langchain_core.messages import AnyMessage,SystemMessage,HumanMessage,ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

_search_wrapper=DuckDuckGoSearchAPIWrapper(max_results=4)
tool=DuckDuckGoSearchResults(api_wrapper=_search_wrapper)
print(type(tool))
print(tool.name)

class AgentState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]

class Agent:
    def __init__(self,model,tools,system=""):
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
        self.graph=graph.compile()
        self.tools={t.name:t for  t in tools}
        self.model=model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result=state["messages"][-1]
        return len(result.tool_calls)>0
    def call_openai(self,state:AgentState):
        messages=state["messages"]
        if self.system:
            messages=[SystemMessage(content=self.system)]+messages
        message=self.model.invoke(messages)
        return {"messages":[message]}
    def take_action(self,state: AgentState):
        tool_calls=state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t["name"],content=str(result)))
        print("Back to the model!")
        return {'messages': results}


prompt = """你是一个聪明的研究助手。请使用搜索引擎来查找信息。\
你可以多次调用搜索（可以一次并行发起多个，也可以分几轮依次调用）。\
只在你清楚自己要查什么的时候才去搜索。\
如果在追问之前需要先查一些信息，你也可以先查再问！请用中文回答最终结果。
"""

model=ChatOpenAI(api_key=OLLAMA_API_KEY,model=MODEL,base_url=OLLAMA_BASE_URL)
abot=Agent(model,[tool],system=prompt)
print(abot.graph.get_graph().draw_mermaid())
messages = [HumanMessage(content="旧金山现在天气怎么样？")]
result = abot.graph.invoke({"messages": messages})

print(result["messages"][-1].content)

query = "2024 年的超级碗（Super Bowl）是谁赢的？夺冠球队的总部位于哪个州？该州的 GDP 是多少？请分别回答每个问题。"
messages=[HumanMessage(content=query)]
model = ChatOpenAI(model=MODEL, base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
abot=Agent(model,[tool],system=prompt)
print(result["messages"][-1].content)




tool=TavilySearchResults(max_results=4)

class AgentStatu(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]

class Agent:
    def __init__(self,model,tools,system=""):
        self.system=system
        graph=StateGraph(AgentStatu)
        graph.add_node("llm",self.call_openai)
        graph.add_node("action",self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True:"action",False:END}
        )
        graph.add_edge("action","llm")
        graph.set_entry_point("llm")
        self.graph=graph.compile()
        self.tools={t.name : t for t in tools}
        self.model=model.bind_tools(tools)

    def exists_action(self,state:AgentStatu):
        result=state["messages"][-1]
        return len(result.tool_calls)>0
    def call_openai(self,state:AgentStatu):
        messages = state['messages']
        if self.system:
            messages=[SystemMessage(content=self.system)]+messages
        message=self.model.invoke(messages)
        return {"messages":[message]}

    def take_action(self, state: AgentStatu):
        tools=state["messages"][-1].tool_calls
        results=[]
        for  t in tools:
            if not t["name"] in self.tools:
                result = "bad tool name, retry"
            else:
                result=self.tools[t["name"]].invoke(t["args"])
            results.append(ToolMessage(tool_call_id=t["id"],name=t["name"],content=str(result)))

        return {"messages":results}

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

model=ChatOpenAI(model="gpt-3.5-turbo")
abot=Agent(model,[tool],system=prompt)
print(abot.graph.get_graph().draw_mermaid())
messages = [HumanMessage(content="What is the weather in sf?")]
result=abot.graph.invoke({"messages":messages})
result['messages'][-1].content
messages = [HumanMessage(content="What is the weather in SF and LA?")]
result = abot.graph.invoke({"messages": messages})
query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
What is the GDP of that state? Answer each question."
messages = [HumanMessage(content=query)]

model = ChatOpenAI(model="gpt-4o")
abot = Agent(model, [tool], system=prompt)
result = abot.graph.invoke({"messages": messages})
print(result['messages'][-1].content)
