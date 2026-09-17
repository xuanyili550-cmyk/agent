import openai
import re
import httpx
import os

from dotenv import load_dotenv
_=load_dotenv()

from openai import OpenAI

client = OpenAI()   # 从 .env 里的 OPENAI_API_KEY 自动读取密钥，别再把字面量字符串当 key
client_completions=client.chat.completions.create(
    model='gpt-4o',   # 原来的 'gpt-5.6-luna' 是不存在的模型，会 404
    messages=[{"role": "user", "content": "Hello world"}]
)
print(client_completions.choices[0].message.content)


class Agent:
    def __init__(self,system=""):
        self.system=system
        self.messages=[]
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def  __call__(self,message):
        self.messages.append({"role":"user","content":message})
        result=self.execute()
        self.messages.append({"role":"assistant","content":result})
        return result   # 少了这行会让 result=abot(...) 拿到 None，后面 result.split() 直接崩


    def execute(self):
        completion=client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=self.messages
        )
        return completion.choices[0].message.content


prompt="""
你需要按照“思考 (Thought)、行动 (Action)、暂停 (PAUSE)、观察 (Observation)”的循环流程进行操作。
在循环结束时，你需要输出一个“答案 (Answer)”。
使用“思考”来描述你对所提问题的想法。
使用“行动”来执行可用的操作之一，然后返回“暂停”。
“观察”即为执行这些操作后的结果。

可用的操作包括：

calculate（计算）：
例如：calculate: 4 * 7 / 3
执行计算并返回数值——使用 Python 语法，因此必要时请使用浮点数格式。

average_dog_weight（平均犬重）：
例如：average_dog_weight: Collie
根据犬种名称返回该品种犬的平均体重。

交互示例：

问题：斗牛犬（Bulldog）的体重是多少？
思考：我应该使用 average_dog_weight 来查询该犬种的体重。
行动：average_dog_weight: Bulldog
暂停

随后你将收到如下输入：

观察：斗牛犬的体重为 51 磅。

接着你输出：

答案：斗牛犬的体重为 51 磅。
""".strip()


def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name in "Scottish Terrier":
        return ("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return ("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return ("a toy poodles average weight is 7 lbs")
    else:
        return ("An average dog weights 50 lbs")

know_actions={
    "calculate":calculate,
    "average_dog_weight":average_dog_weight
}
abot=Agent(prompt)

result=abot("how much does a toy poodle weigh?")
print(result)

result=(average_dog_weight("Toy Poodle"))
print(result)

next_prompt="Observation:{}".format(result)
print(next_prompt)

abot(next_prompt)

print(abot.messages)

abot=Agent(prompt)

question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""

abot(question)

next_prompt = "Observation: {}".format(average_dog_weight("Border Collie"))
print(next_prompt)

abot(next_prompt)

next_prompt = "Observation: {}".format(average_dog_weight("Scottish Terrier"))
print(next_prompt)

abot(next_prompt)

next_prompt = "Observation: {}".format(eval("37 + 20"))
print(next_prompt)
abot(next_prompt)

action_re = re.compile(r'^Action: (\w+): (.*)$')

def query(question,max_turns=5):
    i=0
    bot=Agent(prompt)
    next_prompt=question
    while i<max_turns:
        i+=1
        result=bot(next_prompt)
        print(result)
        actions=[
            action_re.match(a)
            for a in result.split("\n")
            if action_re.match(a)
        ]
        if actions:
            action,action_input=actions[0].groups()
            if action not in know_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation=know_actions[action](action_input)
            print("Observation:", observation)
            next_prompt="Observation:{}".format(observation)
        else:
            return

question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
query(question)



import  openai
import os
import httpx
from dotenv import load_dotenv
_=load_dotenv()
from openai import OpenAI
client=OpenAI(base_url="http://localhost:11434/v1", api_key="ollama",)
MODEL='qwen2.5'
chat_completion=client.chat.completions.create(
    model=MODEL,
    messages=[{"role":"user","content":"你好，请用一句话中文自我介绍。"}]
)
print(chat_completion.choices[0].message)

class Agent:
    def __init__(self,system=""):
        self.system=system
        self.messages=[]
        if self.system:
            self.messages.append({"role":"system","content":system})

    def __call__(self,messages):
        self.messages.append({"role":"system","content":messages})
        result=self.execute()
        self.messages.append({"role":"assistant","content":result})
        return result

    def execute(self):
        completion=client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=self.messages
        )
        return completion.choices[0].message.content

prompt = """
你运行在一个「Thought（思考）、Action（行动）、PAUSE（暂停）、Observation（观察）」的循环里。
循环结束时，你输出一个 Answer（最终答案）。
用 Thought 描述你对所提问题的思考。
用 Action 调用你可用的工具之一——然后立即输出 PAUSE 并停止。
Observation 将是运行该工具后得到的结果。

注意：为了让外部程序能稳定解析，Thought/Action/PAUSE/Observation/Answer 这些关键字以及
「Action: 工具名: 工具输入」这一行请保持这种固定格式（关键字用英文，内容可用中文）。

你可用的工具（actions）有：

calculate:
例如 calculate: 4 * 7 / 3
用 Python 执行一段算术表达式并返回数字（必要时用浮点写法）。

average_dog_weight:
例如 average_dog_weight: 边境牧羊犬
给定犬种名称，返回该犬种的平均体重。

一个完整的示例对话：

Question: 斗牛犬有多重？
Thought: 我应该用 average_dog_weight 工具查一下这个犬种的体重
Action: average_dog_weight: 斗牛犬
PAUSE

（你会被再次调用，并收到：）

Observation: 斗牛犬的平均体重约为 51 磅

然后你输出：

Answer: 斗牛犬的平均体重约为 51 磅
""".strip()

def calculate(what):
    return eval(what)
def average_dog_weight(name):
    if name in "苏格兰梗":
        return ("苏格兰梗的平均体重约为 20 磅")
    elif name in "边境牧羊犬":
        return ("边境牧羊犬的平均体重约为 37 磅")
    elif name in "玩具贵宾犬":
        return ("玩具贵宾犬的平均体重约为 7 磅")
    else:
        return ("一只普通的狗平均体重约为 50 磅")

know_actions={
    "calculate":calculate,
    "average_dog_weight":average_dog_weight
}

abot=Agent(prompt)
result = abot("玩具贵宾犬的体重是多少？")
print(result)

result=average_dog_weight("玩具贵宾犬")
print(result)

next_prompt = "Observation: {}".format(result)

abot(next_prompt)
abot.messages
abot=Agent(prompt)

question = "我有两只狗，一只边境牧羊犬和一只苏格兰梗。它们加起来一共多重？"
abot(question)
next_prompt = "Observation: {}".format(average_dog_weight("边境牧羊犬"))
print(next_prompt)

abot(next_prompt)
next_prompt = "Observation: {}".format(average_dog_weight("苏格兰梗"))
print(next_prompt)

abot(next_prompt)

next_prompt = "Observation: {}".format(eval("37 + 20"))

print(next_prompt)

abot(next_prompt)

action_re = re.compile(r'^Action: (\w+): (.*)$')

def query(question,max_turns=5):
    i=0
    bot=Agent(prompt)
    next_prompt=question
    if i<max_turns:
        result=bot(next_prompt)
        print(result)
        actions=[
            action_re.match(a)
            for a in result.split("\n")
            if action_re.match(a)
        ]
        if actions:
            action,action_input=actions[0].groups()
            if action not in know_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation =know_actions[action](action_input)
            print("Observation:", observation)
            next_prompt="Observation:{}".format(observation)   # 原来带大括号会变成 set，不是字符串
        else:
            return

question = "我有两只狗，一只边境牧羊犬和一只苏格兰梗。它们加起来一共多重？"
print(query(question))










