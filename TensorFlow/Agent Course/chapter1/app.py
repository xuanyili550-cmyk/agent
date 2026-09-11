from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, load_tool, tool
from smolagents import FinalAnswerTool, GradioUI
import datetime
import requests
import pytz
import yaml
@tool
def my_custom_tool(arg1: str, arg2: int) -> str:  # 下面是一个什么都不做的工具示例。尽情发挥你的创造力吧！
    # 指定返回类型很重要# 工具描述/参数描述请保持此格式，但你可以随意修改工具"""一个什么都不做的工具参数
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
        tz = pytz.timezone(timezone)
        # 获取该时区的当前时间
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()
model = InferenceClientModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)

# 从 Hub 导入工具
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# 从 prompt.yaml 文件加载系统提示符
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[final_answer],  # 在这里添加你的工具（不要删除 final_answer）
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates  # 将系统提示传递给 CodeAgent
)

GradioUI(agent).launch()