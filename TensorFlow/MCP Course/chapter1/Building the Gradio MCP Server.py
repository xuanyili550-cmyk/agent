"""
================================================================================
 MCP Course · Chapter 1 · 构建 Gradio MCP 服务器（教程片段合集 + 中文逐段讲解）
================================================================================
 ⚠ 这是“教程片段合集”，不是一键能跑的完整 App——里面把多个知识点各写一小段，还有
   重复 import、TODO 占位、远程 URL、需要 HF_TOKEN/SLACK_WEBHOOK 的部分。当“知识点清单”读。
   能真跑的完整版见：../chapter2/chapter2_从零实现MCP服务器.py(手写协议)、
   ../../实战练习/mcp实战/案例6_FastMCP官方SDK生产版.py(官方 FastMCP 真生产版)。

 本文件依次演示 8 段(下面各段有 ── 段N ── 分隔和讲解)：
   段1 最简 Gradio MCP 服务器   段2 Gradio 当 MCP 客户端(连远程)   段3 完整 Agent+MCP
   段4 HF Hub 微型 Agent        段5 git 工具(diff截断)            段6 Slack 通知工具
   段7 MCP 提示模板(@mcp.prompt) 段8 ★完整 FastMCP 服务器(HF打标签机器人)

 核心概念：
   · Gradio 的 launch(mcp_server=True)：把界面函数自动暴露成 MCP 工具(签名+docstring→schema)。
   · @mcp.tool / @mcp.prompt / @mcp.resource：FastMCP 用装饰器把函数变成 MCP 的工具/提示/资源。
   · docstring + 类型提示至关重要——自动生成工具的 name/description/inputSchema 供 AI 发现调用。
================================================================================
"""
# ── 段1 · 最简 Gradio MCP 服务器 ──────────────────────────────────────────────
# 只要“一个带类型提示+docstring 的函数 + gr.Interface + launch(mcp_server=True)”，
# 就同时得到 ①一个网页 ②一个 MCP 服务器(函数自动变成 AI 能调用的工具)。这是最快的方式。
import json
import subprocess
import subprocess

import gradio as gr
import mcp
from gradio import Interface
from textblob import TextBlob


def sentiment_analysis(text: str) -> str:
    """
        Analyze the sentiment of the given text.

        Args:
            text (str): The text to analyze

        Returns:
            str: A JSON string containing polarity, subjectivity, and assessment
        """
    blob = TextBlob(text)
    sentiment = blob.sentiment
    result = {
        # -1（负面）到 1（正面）
        "polarity": round(sentiment.polarity, 2),
        # 0（客观）到 1（主观）
        "subjectivity": round(sentiment.subjectivity, 2),
        "assessment": "positive" if sentiment.polarity > 0 else
        "negative" if sentiment.polarity < 0 else "neutral"
    }
    return json.dumps(result)
    # 创建 Gradio 接口


demo = gr = Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder='Enter text to analyze...'),
    # 已从 gr.JSON() 更改为 gr.Textbox()
    outputs=gr.Textbox(),
    title="Text Sentiment Analysis",
    description="Analyze the sentiment of text using TextBlob"
)
# 则启动界面和 MCP 服务器：
if __name__ == "__main__":
    demo.launch(mcp_server=True)

#
# 让我们来分析一下关键组成部分：
#
# 函数定义：
#
# 该sentiment_analysis函数接收一个文本输入，并返回 JSON 字典的字符串表示形式。
# 它使用 TextBlob 来分析情感。
# 文档字符串至关重要，因为它有助于 Gradio 生成 MCP 工具架构。
# 类型提示（str和dict）有助于定义输入/输出模式
# Gradio界面：
#
# gr.Interface创建 Web 用户界面和 MCP 服务器
# 该功能已自动作为 MCP 工具公开。
# 输入和输出组件定义了该工具的架构。
# JSON 输出组件确保正确的序列化
# MCP 服务器：
#
# 设置mcp_server=True启用 MCP 服务器
# 服务器将于以下时间可用：http://localhost:7860/gradio_api/mcp/sse
# ── 段2 · 反过来：在代码里当 MCP 客户端，连“远程 MCP 服务器” ──────────────────
# 上面是“做服务器”，这里是“做客户端”：用 smolagents 的 MCPClient 连一个远程 MCP 服务器
# (HF Space 上的)，连上后就能列出它暴露的工具(t.name/t.description)。transport="sse"=走 HTTP。
from smolagents.mcp_client import MCPClient

with MCPClient(
        {"url": "https://abidlabs-mcp-tool-http.hf.space/gradio_api/mcp/sse", "transport": "sse", }
) as tools:
    # 远程服务器上的工具可用
    print("\n".join(f"{t.name}: {t.description}" for t in tools))
import gradio as gr
import os

from mcp import StdioServerParameters
from smolagents import InferenceClientModel, CodeAgent, ToolCollection, MCPClient

# 这是我们在上一节中创建的 MCP 客户端
mcp_client = MCPClient(
    {"url": "https://abidlabs-mcp-tool-http.hf.space/gradio_api/mcp/sse", "transport": "sse", }
)
tools = mcp_client.get_tools()

model = InferenceClientModel(token=os.getenv("HF_TOKEN"))
agent = CodeAgent(tools=[*tools], model=model)
demo = gr.ChatInterface(
    fn=lambda message, history: str(agent.run(message)),
    type="messages",
    examples=["Prime factorization of 68"],
    title="Agent with MCP Tools",
    description="This is a simple agent that uses MCP tools to answer questions."
)

demo.launch()

# ── 段3 · 完整例子：把 MCP 工具交给 Agent，用聊天界面对话 ─────────────────────
# CodeAgent 拿到远程 MCP 的工具后，用户在 ChatInterface 里问问题，Agent 自动决定调哪个工具。
# 用 try/finally 保证结束时 mcp_client.disconnect() 断开连接(生产里要清理连接，别泄漏)。
# 完整示例
import gradio as gr
import os
from smolagents import InferenceClientModel, CodeAgent, MCPClient

try:
    mcp_client = MCPClient(
        {"url": "https://abidlabs-mcp-tool-http.hf.space/gradio_api/mcp/sse", "transport": "sse", }
    )
    tools = mcp_client.get_tools()
    model = InferenceClientModel(token=os.getenv("HUGGINGFACE_API_TOKEN"))
    agent = CodeAgent(tools=[*tools], model=model, additional_authorized_imports=["json", "ast", "urllib", "base64"])
    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        type='messages',
        examples=["Analyze the sentiment of the following text 'This is awesome'"],
        title="Agent with MCP Tools",
        description="This is a simple agent that uses MCP tools to answer questions.",
    )
    demo.launch()
finally:
    mcp_client.disconnect()

# ── 段4 · 另一种客户端：huggingface_hub 自带的 Agent ────────────────────────
# huggingface_hub.Agent 也能连 MCP 服务器(servers 里配 command/args，用 npx mcp-remote 桥接)，
# 指定一个大模型(Qwen)当大脑。这是“微型 Agent”的写法，和段3的 smolagents 是两套等价方案。
# 使用 MCP 和 Hugging Face Hub 构建微型代理
# 自定义微型代理 MCP 客户端
import os
from huggingface_hub import Agent

agent = Agent(
    model="Qwen/Qwen2.5-72B-Instruct",
    provider="nebius",
    servers=[
        {
            "command": "npx",
            "args": [
                "mcp-remote",
                "http://localhost:7860/gradio_api/mcp/sse"
            ]
        }
    ],
)


# ── 段5 · 怎么写 MCP 工具(@mcp.tool) + 一个真实 git 工具 ─────────────────────
# @mcp.tool 装饰器把函数变成 MCP 工具：函数签名(参数+类型)→inputSchema，docstring→description，
# 这些是 AI 决定“调不调、怎么传参”的依据，所以 docstring/类型提示必须写清楚。
# 下面 tool_name 是模板；analyze_file_changes 是真实例子：跑 git diff，diff 太长就智能截断(省 token)。
@mcp.tool()
async def tool_name(param1: str, param2: bool = True) -> str:
    """Tool description for Claude.（工具描述——AI 靠它判断这个工具干嘛、要不要用）

    Args:
        param1: Description of parameter
        param2: Optional parameter with default
    """
    # Your implementation
    result = {"key": "value"}
    return json.dumps(result)


@mcp.tool()
async def analyze_file_changes(base_branch: str = "main",
                               include_diff: bool = True,
                               max_diff_lines: int = 500) -> str:
    """Analyze file changes with smart output limiting.

    Args:
        base_branch: Branch to compare against
        include_diff: Whether to include the actual diff
        max_diff_lines: Maximum diff lines to include (default 500)
    """
    try:
        # Get the diff
        result = subprocess.run(
            ["git", "diff", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True
        )

        diff_output = result.stdout
        diff_lines = diff_output.split('\n')

        # Smart truncation if needed
        if len(diff_lines) > max_diff_lines:
            truncated_diff = '\n'.join(diff_lines[:max_diff_lines])
            truncated_diff += f"\n\n... Output truncated. Showing {max_diff_lines} of {len(diff_lines)} lines ..."
            diff_output = truncated_diff

        # Get summary statistics
        stats_result = subprocess.run(
            ["git", "diff", "--stat", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True
        )

        return json.dumps({
            "stats": stats_result.stdout,
            "total_lines": len(diff_lines),
            "diff": diff_output if include_diff else "Use include_diff=true to see diff",
            "files_changed": self._get_changed_files(base_branch)
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def analyze_file_changes(max_diff_lines: int = 500):   # 修复：占位符 (...) 改成合法签名
    # Get Claude's working directory from roots
    context = mcp.get_context()
    roots_result = await context.session.list_roots()

    # Extract the path from the FileUrl object
    working_dir = roots_result.roots[0].uri.path

    # Use it for all git commands
    result = subprocess.run(
        ["git", "diff", "--name-status"],
        capture_output=True,
        text=True,
        cwd=working_dir  # Run in Claude's directory!
    )


# 修复：下面是文档里的“示例片段”，被误粘到模块级(await 不能在函数外)，注释掉以消除语法错：
# context = mcp.get_context()
# roots_result = await context.session.list_roots()
# working_dir = roots_result.roots[0].uri.path  # FileUrl object has .path property
# subprocess.run(["git", "diff"], cwd=working_dir)

# ── 段6 · 另一个工具：Slack 通知(含 TODO 占位) ────────────────────────────────
# 演示工具可以“执行真实副作用”(往 Slack 发消息)。里面用 os.getenv 取 webhook(密钥别硬编码，走环境变量)；
# 具体发送逻辑留了 TODO 占位(课程练习)。生产里工具能删库/花钱，务必校验参数+最小权限。
# Slack 通知
import os
import requests
from mcp.types import TextContent


@mcp.tool()
def send_slack_notification(message: str) -> str:
    """Send a formatted notification to the team Slack channel."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return "Error: SLACK_WEBHOOK_URL environment variable not set"

    try:
        # TODO:向 webhook_url 发送 POST 请求
        # TODO:在 JSON 有效负载中包含消息，并设置 "mrkdwn": true
        # TODO:处理响应并返回状态
        pass
    except Exception as e:
        return f"Error sending message: {str(e)}"


# ── 段7 · MCP 提示(@mcp.prompt)：预定义模板，指导 AI 怎么产出 ────────────────
# 工具是“AI 调的函数”，提示(prompt)是“预定义模板/工作流”。这里定义 CI 失败/成功的 Slack 消息模板，
# AI 套用它就能产出格式统一的通知。客户端用 prompts/list 发现、prompts/get 取渲染后的内容。
# 创建格式提示（15 分钟）
@mcp.prompt()
def format_ci_failure_alert() -> str:
    """Create a Slack alert for CI/CD failures."""
    return """Format this GitHub Actions failure as a Slack message:

请使用以下模板：
:rotating_light: *CI 故障警报* :rotating_light:

CI 工作流失败：
*工作流程*：工作流程名称
*分支*：分支名称
状态：失败
*查看详情*：<LOGS_LINK|查看日志>

请检查日志并解决任何问题。

请使用 Slack Markdown 格式，并保持简洁，以便团队快速浏览。"""


@mcp.prompt()
def format_ci_success_summary() -> str:
    """Create a Slack message celebrating successful deployments."""
    return """将此次成功的 GitHub Actions 运行格式化为 Slack 消息：

请使用以下模板：
:white_check_mark: *部署成功* :white_check_mark:

[存储库名称] 的部署已成功完成。

*变更：*
- 主要功能或修复 1
- 主要功能或修复 2

*链接：*
<PR_LINK|查看更改>

保持庆祝的氛围，但也要兼顾信息量。请使用 Slack Markdown 格式。"""


# 提示符使用示例
# 当测试失败时，Claude 会使用“分析 CI 结果”提示符：
prompt_data = {
    "event_type": "workflow_run",
    "status": "failure",
    "failed_jobs": ["unit-tests", "lint"],
    "error_logs": "...",
    "pr_context": {...}
}

# Claude 生成：
# - 根本原因分析
# - 建议的修复方案
# - 影响评估
# - 后续步骤

# ── 段8 · ★完整 FastMCP 服务器：HF 打标签机器人 ──────────────────────────────
# 这才是一个“像样的完整服务器”：用官方 FastMCP 建服务器，暴露两个工具——
#   get_current_tags(repo_id)：查一个 HF 模型仓库当前的标签
#   add_new_tag(repo_id, new_tag)：给仓库加标签(通过创建 PR 的方式，安全、可审阅)
# 生产要点都体现了：没配 token 时优雅返回错误、try/except 包住外部调用、返回结构化 JSON。
# 这是本文件最值得照着学的一段(能真跑，需 HF_TOKEN)。
# 完成 MCP 服务器实施
# !/usr/bin/env python3
"""
Simplified MCP Server for HuggingFace Hub Tagging Operations using FastMCP
"""

import os
import json
from fastmcp import FastMCP
from huggingface_hub import HfApi, model_info, ModelCard, ModelCardData
from huggingface_hub.utils import HfHubHTTPError
from dotenv import load_dotenv

load_dotenv()
# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize HF API client
hf_api = HfApi(token=HF_TOKEN) if HF_TOKEN else None

# Create the FastMCP server
mcp = FastMCP("hf-tagging-bot")


@mcp.tool()
def get_current_tags(repo_id: str) -> str:
    """Get current tags from a HuggingFace model repository"""
    print(f"🔧 get_current_tags called with repo_id: {repo_id}")

    if not hf_api:
        error_result = {"error": "HF token not configured"}
        json_str = json.dumps(error_result)
        print(f"❌ No HF API token - returning: {json_str}")
        return json_str

    try:
        print(f"📡 Fetching model info for: {repo_id}")
        info = model_info(repo_id=repo_id, token=HF_TOKEN)
        current_tags = info.tags if info.tags else []
        print(f"🏷️ Found {len(current_tags)} tags: {current_tags}")

        result = {
            "status": "success",
            "repo_id": repo_id,
            "current_tags": current_tags,
            "count": len(current_tags),
        }
        json_str = json.dumps(result)
        print(f"✅ get_current_tags returning: {json_str}")
        return json_str

    except Exception as e:
        print(f"❌ Error in get_current_tags: {str(e)}")
        error_result = {"status": "error", "repo_id": repo_id, "error": str(e)}
        json_str = json.dumps(error_result)
        print(f"❌ get_current_tags error returning: {json_str}")
        return json_str


@mcp.tool()
def add_new_tag(repo_id: str, new_tag: str) -> str:
    """通过 PR 向 HuggingFace 模型仓库添加新标签"""
    print(f"🔧 add_new_tag called with repo_id: {repo_id}, new_tag: {new_tag}")

    if not hf_api:
        error_result = {"error": "HF token not configured"}
        json_str = json.dumps(error_result)
        print(f"❌ No HF API token - returning: {json_str}")
        return json_str                 # 修复：原本漏了 return

    try:
        # 获取当前模型信息和标签
        info = model_info(repo_id=repo_id, token=HF_TOKEN)
        current_tags = info.tags if info.tags else []
        print(f"🏷️ Current tags: {current_tags}")

        # 标签已存在则直接返回
        if new_tag in current_tags:
            return json.dumps({"status": "already_exists", "repo_id": repo_id,
                               "tag": new_tag, "message": f"Tag '{new_tag}' already exists"})

        updated_tags = current_tags + [new_tag]
        print(f"🆕 Update tags {current_tags} -> {updated_tags}")

        # 加载或新建模型卡片，更新标签
        try:
            card = ModelCard.load(repo_id, token=HF_TOKEN)
            if not hasattr(card, "data") or card.data is None:
                card.data = ModelCardData()
        except HfHubHTTPError:
            card = ModelCard("")
            card.data = ModelCardData()
        card_dict = card.data.to_dict()
        card_dict["tags"] = updated_tags
        card.data = ModelCardData(**card_dict)

        # 创建包含更新模型卡的 PR
        from huggingface_hub import CommitOperationAdd
        pr_title = f"Add '{new_tag}' tag"
        commit_info = hf_api.create_commit(
            repo_id=repo_id,
            operations=[CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=str(card).encode("utf-8"))],
            commit_message=pr_title, create_pr=True, token=HF_TOKEN)
        pr_url = getattr(commit_info, "pr_url", str(commit_info))
        return json.dumps({"status": "success", "repo_id": repo_id, "tag": new_tag,
                           "pr_url": pr_url, "previous_tags": current_tags,
                           "new_tags": updated_tags,
                           "message": f"Created PR to add tag '{new_tag}'"})
    except Exception as e:               # 修复：统一的外层 except(原代码缩进坍了)
        import traceback
        print(f"❌ Error in add_new_tag: {e}")
        return json.dumps({"status": "error", "repo_id": repo_id, "tag": new_tag,
                           "error": str(e), "traceback": traceback.format_exc()})
