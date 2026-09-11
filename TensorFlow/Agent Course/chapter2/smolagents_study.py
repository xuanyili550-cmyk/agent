# 合成剂中的模型集成
# smolagents支持灵活的LLM集成，允许您使用任何符合特定条件的可调用模型。该框架提供了几个预定义的类来简化模型连接：
#
# TransformersModel：实现本地transformers管道以实现无缝集成。
# InferenceClientModel：支持通过Hugging Face 的基础架构或通过越来越多的第三方推理提供程序进行无服务器推理调用。
# LiteLLMModel：利用LiteLLM实现轻量级模型交互。
# OpenAIServerModel：连接到任何提供 OpenAI API 接口的服务。
# AzureOpenAIServerModel：支持与任何 Azure OpenAI 部署集成。

#运行代理程序非常简单
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, tool

agent=CodeAgent(tools=[DuckDuckGoSearchTool()],model=InferenceClientModel())
agent.run( "为韦恩豪宅的派对搜索最佳音乐推荐。" )

# 根据场合推荐菜单的工具
@tool
def suggest_menu(occasion:str)->str:
    """
        Suggests a menu based on the occasion.
        Args:
            occasion (str): The type of occasion for the party. Allowed values are:
                            - "casual": Menu for casual party.
                            - "formal": Menu for formal party.
                            - "superhero": Menu for superhero party.
                            - "custom": Custom menu.
        """
    if occasion=="casual":
        return "Pizza, snacks, and drinks."
    elif occasion=="formal":
        return "3-course dinner with wine and dessert."
    elif occasion=="superhero":
        return "Buffet with high-energy and healthy food."
    elif occasion=="custom":
        return "Custom menu for the butler."
    else:
        return "err"
    # 管家阿尔弗雷德正在为宴会准备菜单
    # 正在为宴会准备菜单
agent=CodeAgent(tools=[suggest_menu],model=InferenceClientModel())
agent.run("Prepare a formal menu for the party.")

#我们将使用它additional_authorized_imports来允许导入datetime模块
from smolagents import CodeAgent, InferenceClientModel
import numpy as np
import time
import datetime
agent = CodeAgent(tools=[], model=InferenceClientModel(), additional_authorized_imports=['datetime'])
agent.run("""
    阿尔弗雷德需要为派对做准备。以下是任务：
    1. 准备饮料 - 30 分钟
    2. 装饰豪宅 - 60 分钟
    3. 设置菜单 - 45 分钟
    4. 准备音乐和播放列表 - 45 分钟
    如果我们现在开始，派对什么时候准备好？
    """
          )
#您可以在smolagents 文档中了解更多关于如何构建代码代理的信息。
# Change to your username and repo name
agent.push_to_hub('sergiopaniego/AlfredAgent')
# Change to your username and repo name
alfred_agent = agent.from_hub('sergiopaniego/AlfredAgent', trust_remote_code=True)

alfred_agent.run("Give me the best playlist for a party at Wayne's mansion. The party idea is a 'villain masquerade' theme")

from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, Tool, tool, \
    VisitWebpageTool


@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion: The type of occasion for the party.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."


@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.

    Args:
        query: A search term for finding catering services.
    """
    # 餐饮服务及其评分示例列表
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }

    # 查找评分最高的餐饮服务（模拟搜索查询过滤）
    best_service = max(services, key=services.get)

    return best_service


class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""

    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
        }
    }

    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }

        return themes.get(category.lower(),
                          "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")


# 管家阿尔弗雷德正在准备派对菜单
agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        suggest_menu,
        catering_service_tool,
        SuperheroPartyThemeTool(),
        FinalAnswerTool()
    ],
    model=InferenceClientModel(),
    max_steps=10,
    verbosity_level=2
)

agent.run(
    "Give me the best playlist for a party at the Wayne's mansion. The party idea is a 'villain masquerade' theme")


#使用 OpenTelemetry 和 Langfuse 检查我们的派对准备代理
import os

# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region

from langfuse import get_client

langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")


from openinference.instrumentation.smolagents import SmolagentsInstrumentor
SmolagentsInstrumentor().instrument()

from smolagents import CodeAgent,InferenceClientModel
agent=CodeAgent(tools=[],model=InferenceClientModel())
alfred_agent = agent.from_hub( 'sergiopaniego/AlfredAgent' , trust_remote_code= True )
alfred_agent.run( "给我一个适合在韦恩豪宅举办的派对的最佳歌单。派对主题是“反派化装舞会”。" )

#运行工具调用代理
from smolagents import ToolCallingAgent,WebSearchTool,InferenceClientModel
agent=ToolCallingAgent(tools=[WebSearchTool()],model=InferenceClientModel())
agent.run( "为韦恩豪宅的派对搜索最佳音乐推荐。" )

#Alfred 如何使用@tool装饰器来实现这一点
from smolagents import CodeAgent, InferenceClientModel, tool
# 假设我们有一个函数，用于获取评分最高的餐饮服务。
@tool
def catering_service_tool(query:str)->str:
    """
        This tool returns the highest-rated catering service in Gotham City.

        Args:
            query: A search term for finding catering services.
        """
    # 餐饮服务及其评分示例列表
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }
    # 查找评分最高的餐饮服务（模拟搜索查询过滤）
    best_service=max(services,key=services.get)
    return best_service
agent=CodeAgent(tools=[catering_service_tool],model=InferenceClientModel())
# 运行代理以查找最佳餐饮服务
result = agent.run(
    "Can you give me the name of the highest-rated catering service in Gotham City?"
)
print(result)


from smolagents import Tool, CodeAgent, InferenceClientModel

class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""

    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
        }
    }

    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }

        return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")

# 实例化工具
party_theme_tool = SuperheroPartyThemeTool()
agent = CodeAgent(tools=[party_theme_tool], model=InferenceClientModel())

# 运行代理以生成派对主题创意
result = agent.run(
    "What would be a good superhero party idea for a 'villain masquerade' theme?"
)

print(result)  # 输出：“哥谭恶棍舞会：一场神秘的化装舞会，宾客们装扮成经典的蝙蝠侠反派。”

from smolagents import load_tool, CodeAgent, InferenceClientModel

image_generation_tool = load_tool(
    "m-ric/text-to-image",
    trust_remote_code=True
)

agent = CodeAgent(
    tools=[image_generation_tool],
    model=InferenceClientModel()
)

agent.run("Generate an image of a luxurious superhero-themed party at Wayne Manor with made-up superheros.")


from smolagents import CodeAgent, InferenceClientModel, Tool

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct")

agent = CodeAgent(tools=[image_generation_tool], model=model)

agent.run(
    "Improve this prompt, then generate an image of it.",
    additional_args={'user_prompt': 'A grand superhero-themed party at Wayne Manor, with Alfred overseeing a luxurious gala'}
)

#langchain 集成后
# langchain v1 已把 load_tools 从 langchain.agents 移到 langchain_community
from langchain_community.agent_toolkits.load_tools import load_tools
from smolagents import CodeAgent, InferenceClientModel, Tool

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = CodeAgent(tools=[search_tool], model=model)

agent.run("Search for luxury entertainment ideas for a superhero-themed event, such as live performances and interactive experiences.")










