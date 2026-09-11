"""
================================================================================
 MCP Course · Chapter 1 · MCP 介绍（原始课程笔记：四大原语的概念 + 玩具函数）
================================================================================
 这份原始文件是概念说明 + 几个“长得像工具/资源/提示”的玩具函数(不是真 MCP 服务器)。
 能真跑、可上简历的完整版本见本 MCP Course 目录：
   · chapter1/chapter1_MCP协议入门_学习笔记.py   概念 + 真实 JSON-RPC 消息长什么样(可跑)
   · chapter2/chapter2_从零实现MCP服务器.py      ★从零实现真 MCP 服务器+客户端(stdio/JSON-RPC)
   · chapter3/chapter3_资源_提示_传输.py          资源/提示/采样/传输(真协议)
   · chapter4/chapter4_综合实战_Agent与官方SDK.py MCP Agent 循环 + FastMCP 官方 SDK + 集成部署
 本机没装官方 SDK(mcp/fastmcp)，所以“可跑”的部分用纯 Python 手写 JSON-RPC(反而更懂底层)；
 官方 FastMCP 写法在 chapter4 作生产参考(pip install mcp)。
================================================================================
"""
# 能力	描述	例子
# 工具	AI模型可以调用的可执行函数，用于执行操作或检索计算数据。通常与应用程序的使用案例相关。
# 天气应用程序中的一个工具可能是一个返回特定地点天气情况的函数。
# 资源	只读数据源，提供上下文信息，无需进行大量计算。
# 研究助理可能掌握一些科学论文资源。
# 提示	预定义的模板或工作流程，用于指导用户、人工智能模型和可用功能之间的交互。
# 总结提示。
# 采样	服务器发起请求，要求客户端/主机执行 LLM 交互，从而实现递归操作，使 LLM 能够查看生成的内容并做出进一步的决定。
# 一款写作应用程序会检查自己的输出结果，并决定进一步改进。

#标准输入/输出 (stdio)JSON-RPC 定义了消息格式


def get_weather(location: str) -> dict:
    """Get the current weather for a specified location."""
    # 连接到天气 API 并获取数据
    return {
        "temperature": 72,
        "conditions": "Sunny",
        "humidity": 45
    }
def read_file(file_path: str) -> str:
    """Read the contents of a file at the specified path."""
    #读取指定路径下文件的内容
    with open(file_path, 'r') as f:
        return f.read()
#生成代码审查的提示模板
def code_review(code: str, language: str) -> list:
    #为提供的代码片段生成代码审查
    """Generate a code review for the provided code snippet."""
    return [
        {
            "role": "system",
            "content": f"You are a code reviewer examining {language} code. Provide a detailed review highlighting best practices, potential issues, and suggestions for improvement."
        },
        {
            "role": "user",
            "content": f"Please review this {language} code:\n\n```{language}\n{code}\n```"
        }
    ]

#服务器可能会请求客户端分析它已处理的数据
def request_sampling(messages, system_prompt=None, include_context="none"):
    """Request LLM sampling from the client."""
    # 在实际实现中，这将向客户端发送请求
    return {
        "role": "assistant",
        "content": "Analysis of the provided data..."
    }

# MCP 的关键特性之一是动态功能发现。当客户端连接到服务器时，它可以通过特定的列表方法查询可用的工具、资源和提示：
#
# tools/list：发现可用工具
# resources/list：发现可用资源
# prompts/list：发现可用提示

from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务器
mcp = FastMCP("Weather Service")

# 工具实现
@mcp.tool()
def get_weather(location: str) -> str:
    """Get the current weather for a specified location."""
    return f"Weather in {location}: Sunny, 72°F"

# 资源实现
@mcp.resource("weather://{location}")
def weather_resource(location: str) -> str:
    """Provide weather data as a resource."""
    return f"Weather data for {location}: Sunny, 72°F"

# 提示符实现
@mcp.prompt()
def weather_report(location: str) -> str:
    """Create a weather report prompt."""
    return f"""You are a weather reporter. Weather report for {location}?"""


# Run the server
if __name__ == "__main__":
    mcp.run()

# MCP 的设计与语言无关，并且有适用于几种流行编程语言的官方 SDK：
# Language	Repository	Maintainer(s)	Status
# TypeScript	github.com/modelcontextprotocol/typescript-sdk	Anthropic	Active
# Python	github.com/modelcontextprotocol/python-sdk	Anthropic	Active
# Java	github.com/modelcontextprotocol/java-sdk	Spring AI (VMware)	Active
# Kotlin	github.com/modelcontextprotocol/kotlin-sdk	JetBrains	Active
# C#	github.com/modelcontextprotocol/csharp-sdk	Microsoft	Active (Preview)
# Swift	github.com/modelcontextprotocol/swift-sdk	loopwork-ai	Active
# Rust	github.com/modelcontextprotocol/rust-sdk	Anthropic/Community	Active
# Dart	https://github.com/leehack/mcp_dart	Flutter Community	Active
{
    "name": "playwright-agent",
    "description": "Agent with Playwright MCP server",
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "provider": "nebius",
    "servers": [
        {
            "type": "stdio",
            "command": "npx",
            "args": ["@playwright/mcp@latest"]
        }
    ]
}

import gradio as gr

def letter_counter(word: str, letter: str) -> int:
    """
    Count the number of occurrences of a letter in a word or text.

    Args:
        word (str): The input text to search through
        letter (str): The letter to search for

    Returns:
        int: The number of times the letter appears in the text
    """
    word = word.lower()
    letter = letter.lower()
    count = word.count(letter)
    return count

# 创建标准 Gradio 接口
demo = gr.Interface(
    fn=letter_counter,
    inputs=["textbox", "textbox"],
    outputs="number",
    title="Letter Counter",
    description="Enter text and a letter to count how many times the letter appears in the text."
)

# 则启动 Gradio Web 界面和 MCP 服务器：
if __name__ == "__main__":
    demo.launch(mcp_server=True)