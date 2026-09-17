import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
import warnings
warnings.filterwarnings('ignore')

import datetime
current_date=datetime.datetime.now().date()
target_date = datetime.date(2024, 6, 12)
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
llm=ChatOpenAI(model=llm_model,temperature=0.0)
memory=ConversationBufferMemory()
conversation=ConversationChain(
    llm,
    memory=memory,
    verbose=True
)
conversation.predict(input="Hi, my name is Andrew")

conversation.predict(input="What is 1+1?")

conversation.predict(input="What is my name?")
print(memory.buffer)
memory.load_memory_variables({})
memory = ConversationBufferMemory()

memory.save_context({"input": "Hi"},
                    {"output": "What's up"})

print(memory.buffer)
memory.load_memory_variables({})

memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})

memory.load_memory_variables({})

from langchain_classic.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})

memory.load_memory_variables({})


llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=False
)

conversation.predict(input="Hi, my name is Andrew")

conversation.predict(input="What is 1+1?")

conversation.predict(input="What is my name?")


from langchain_classic.memory import ConversationTokenBufferMemory

llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"},
                    {"output": "Charming!"})

memory.load_memory_variables({})

from langchain_classic.memory import ConversationSummaryBufferMemory
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"},
                    {"output": f"{schedule}"})
memory.load_memory_variables({})

conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=True
)
conversation.predict(input="What would be a good demo to show?")
memory.load_memory_variables({})