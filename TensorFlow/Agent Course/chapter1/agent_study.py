from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#通用工具实现
from typing import Callable

# name（字符串）：工具的名称。
# description（字符串）：工具功能的简要描述。
# function（可调用对象）：工具执行的函数。
# arguments（列表）：预期的输入参数。
# outputs（字符串或列表）：工具的预期输出。
# __call__()：当工具实例被调用时调用该函数。
# to_string()将工具的属性转换为文本表示形式。
class Tool:
    """
    A class representing a reusable piece of code (Tool).

    Attributes:
        name (str): Name of the tool.
        description (str): A textual description of what the tool does.
        func (callable): The function this tool wraps.
        arguments (list): A list of arguments.
        outputs (str or list): The return type(s) of the wrapped function.
    """
    def __init__(self,
                 name: str,
                 description: str,
                 func: Callable,
                 arguments: list,
                 outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        """
        Return a string representation of the tool,
        including its name, description, arguments, and outputs.
        """
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])

        return (
            f"Tool Name: {self.name},"
            f" Description: {self.description},"
            f" Arguments: {args_str},"
            f" Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs):
        """
        Invoke the underlying function (callable) with provided arguments.
        """
        return self.func(*args, **kwargs)

calculator_tool = Tool(
    "calculator",                   # name
    "Multiply two integers.",       # description
    calculator,                     # function to call
    [("a", "int"), ("b", "int")],   # inputs (names and types)
    "int",                          # output
)

@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())

#思维链（CoT）是一种引导模型逐步思考问题，最终得出最终答案的技巧
#ReAct：推理 + 行动
# 示例（ReAct）
# 想法：我需要查找巴黎的最新天气。
# 操作：搜索[ “巴黎天气” ]
# 观察：18°C，多云。
# 想法：现在我知道天气了……
# 操作：完成[ “巴黎18°C，多云。” ]
# ReAct 与 CoT
# 特征	思维链（CoT）	反应
# 逐步逻辑	✅ 是的	✅ 是的
# 外部工具	❌ 否	✅ 是的（行动 + 观察）
# 最适合	逻辑、数学、内部任务	信息检索、动态多步骤任务

# 代码代理示例：获取天气信息
def get_weather(city):
    import requests
    api_url = f"https://api.weather.com/v1/location/{city}?apiKey=YOUR_API_KEY"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("weather", "No weather information available")
    else:
        return "Error: Unable to fetch weather data."

# 执行函数并准备最终结果# 执行函数并准备最终结果
result = get_weather("New York")
final_answer = f"The current weather in New York is: {result}"
print(final_answer)

#无服务器 API
import os
from huggingface_hub import InferenceClient
## You need a token from https://hf.co/settings/tokens, ensure that you select 'read' as the token type. If you run this on Google Colab, you can set it up in the "settings" tab under "secrets". Make sure to call it "HF_TOKEN"
# HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(model="moonshotai/Kimi-K2.5")
output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of France is"},
    ],
    stream=False,
    max_tokens=1024,
    extra_body={'thinking': {'type': 'disabled'}},
)
print(output.choices[0].message.content)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
]

print(messages)

output = client.chat.completions.create(
    messages=messages,
    stream=False,
    max_tokens=200,
    extra_body={'thinking': {'type': 'disabled'}},
)
print(output.choices[0].message.content)

# 这个答案是模型产生的幻觉。我们需要停下来，实际执行一下这个函数！
output = client.chat.completions.create(
    messages=messages,
    max_tokens=150,
    stop=["Observation:"], # 在调用任何实际函数之前停止
    extra_body={'thinking': {'type': 'disabled'}},
)

print(output.choices[0].message.content)
# 虚拟函数
def get_weather(location):
    return f"the weather in {location} is sunny with low temperatures. \n"

get_weather('London')

messages=[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London ?"},
    {"role": "assistant", "content": output.choices[0].message.content + "Observation:\n" + get_weather('London')},
]

output = client.chat.completions.create(
    messages=messages,
    stream=False,
    max_tokens=200,
    extra_body={'thinking': {'type': 'disabled'}},
)

print(output.choices[0].message.content)


#使用smolagents中的CodeAgent类
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, load_tool, tool, GradioUI
import datetime
import requests
import pytz
import yaml

@tool
def my_custom_tool(arg1:str,arg2:int)-> str:
    # 指定返回类型很重要
    # 工具描述/参数描述请保持此格式，但您可以随意修改工具
    """A tool that does nothing yet
        Args:
            arg1: the first argument
            arg2: the second argument
        """
    return "What magic will you build ?"
@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
        Args:
            timezone: A string representing a valid timezone (e.g., 'America/New_York').
        """
    try:
        # 创建时区对象
        tz=pytz.timezone(timezone)
        # 获取该时区的当前时间
        local_time=datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

final_answer=FinalAnswerTool()
model=InferenceClientModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)
with open ( "prompts.yaml" , 'r' ) as stream:
    prompt_templates = yaml.safe_load(stream) # 我们正在创建 CodeAgent
agent = CodeAgent(
    model=model,
    tools=[final_answer], # 在这里添加你的工具（不要删除 final_answer）
    max_steps= 6 ,
    verbosity_level= 1 ,
    grammar= None ,
    planning_interval= None ,
    name= None ,
    description= None ,
    prompt_templates=prompt_templates
)
GradioUI(agent).launch()
