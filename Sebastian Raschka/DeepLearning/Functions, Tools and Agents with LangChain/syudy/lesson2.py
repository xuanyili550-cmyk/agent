import json

import openai
import re
import httpx
import os

from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())

from openai import OpenAI
openai.api_key=os.environ["OPENAI_API_KEY"]
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
prompt=ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)
model=ChatOpenAI()
output_parser=StrOutputParser()
# LCEL 的核心语法：用 | 把多个 Runnable 串联成一条 chain
# 数据流向：用户输入 -> prompt（拼成完整的消息）-> model（调用大模型）-> output_parser（解析成字符串）
chain=prompt|model|output_parser
chain.invoke({"topic":"bears"})

vectorstore=Chroma.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings()
)
retriever=vectorstore.as_retriever()
retriever.invoke("where did harrison work?")

retriever.invoke("what do bears like to eat")
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
from langchain_core.runnables import RunnableMap
chain=RunnableMap(
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"]
    }
)|prompt|model|output_parser
chain.invoke({"question": "where did harrison work?"})
inputs = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
})
inputs.invoke({"question": "where did harrison work?"})
functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    }
  ]
prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}")
    ]
)
model = ChatOpenAI(temperature=0).bind(functions=functions)
runnable = prompt | model
runnable.invoke({"input": "what is the weather in sf"})
functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    },
        {
      "name": "sports_search",
      "description": "Search for news of recent sport events",
      "parameters": {
        "type": "object",
        "properties": {
          "team_name": {
            "type": "string",
            "description": "The sports team to search for"
          },
        },
        "required": ["team_name"]
      }
    }
  ]
model=model.bind(functions=functions)
runnable=prompt|model
runnable.invoke({"input": "how did the patriots do yesterday?"})
simple_model = OpenAI(
    temperature=0,
    max_tokens=1000,
    model="gpt-3.5-turbo-instruct"
)
simple_chain = simple_model | json.loads
challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"
simple_model.invoke(challenge)
simple_chain.invoke(challenge)
model = ChatOpenAI(temperature=0)
chain = model | StrOutputParser() | json.loads
chain.invoke(challenge)
final_chain=simple_chain.with_fallbacks([chain])
final_chain.invoke(challenge)
prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser
chain.batch([{"topic": "bears"}, {"topic": "frogs"}])


