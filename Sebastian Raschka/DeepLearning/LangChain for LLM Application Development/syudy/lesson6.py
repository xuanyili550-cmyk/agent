import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
import warnings
warnings.filterwarnings('ignore')

import datetime
current_data=datetime.datetime.now().date()
target_data=datetime.date(2024,6,12)

if current_data>target_data:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_classic.agents import load_tools, initialize_agent
from langchain_classic.agents import AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(model=llm_model,temperature=0.0)
tools = load_tools(["llm-math","wikipedia"], llm=llm)
agent=initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)
agent("What is the 25% of 300?")
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)
customer_list = [["Harrison", "Chase"],
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"],
                 ["Geoff","Fusion"],
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]

agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""")
from langchain_core.globals import set_debug

set_debug(True)
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""")
set_debug(False)

from langchain_core.tools import tool
from datetime import date

@tool
def time(text: str) -> str:
    return str(date.today())

agent= initialize_agent(
    tools + [time],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
try:
    result = agent("whats the date today?")
except:
    print("exception on external access")