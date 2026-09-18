# ⚠️ 原始课程文件（用户手写）。已修正两处 IDE 误加的坏导入：
#    · from ast import Dict            → from typing import Dict（ast.Dict 不是类型，纯属 IDE 乱补）
#    · from urllib.request import Request → from fastapi import Request（这里的 Request 是 FastAPI 的）
#    其余内容保持用户原样。本文件是“HF 打标机器人”的 webhook+Agent 参考实现，依赖真实
#    HF_TOKEN / 运行中的 FastAPI 服务，属集成参考，不作独立运行；可跑的纯 Python 版见 chapter3 案例。
import os
from typing import Dict
from fastapi import Request

from huggingface_hub.inference._mcp.agent import Agent
from typing import Optional, Literal, Any

# 配置
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "microsoft/DialoGPT-medium")
DEFAULT_PROVIDER: Literal["hf-inference"] = "hf-inference"

# 全局代理实例
agent_instance: Optional[Agent] = None
async def get_agent():
    """Get or create Agent instance"""
    print("🤖 get_agent() called...")
    global agent_instance
    if agent_instance is None and HF_TOKEN:
        print("🔧 Creating new Agent instance...")
        print(f"🔑 HF_TOKEN present: {bool(HF_TOKEN)}")
        print(f"🤖 Model: {HF_MODEL}")
        print(f"🔗 Provider: {DEFAULT_PROVIDER}")
        try:
            agent_instance = Agent(
                model=HF_MODEL,
                provider=DEFAULT_PROVIDER,
                api_key=HF_TOKEN,
                servers=[
                    {
                        "type": "stdio",
                        "command": "python",
                        "args": ["mcp_server.py"],
                        "cwd": ".",
                        "env": {"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {},
                    }
                ],
            )
            print("✅ Agent instance created successfully")
            print("🔧 Loading tools...")
            await agent_instance.load_tools()
            print("✅ Tools loaded successfully")
        except Exception as e:
            print(f"❌ Error creating/loading agent: {str(e)}")
            agent_instance = None
    return agent_instance                # 修复:原缺 return,导致调用方 await get_agent() 恒为 None


# 智能体如何使用工具的示例
async def example_tool_usage():
    agent = await get_agent()

    if agent:
        # 智能体可以推断要使用哪些工具
        response = await agent.run(
            "Check the current tags for microsoft/DialoGPT-medium and add the tag 'conversational-ai' if it's not already present"
        )
        print(response)


async def process_webhook_comment(webhook_data: Dict[str, Any]):
    """Process webhook to detect and add tags"""
    print("🏷️ Starting process_webhook_comment...")

    try:
        comment_content = webhook_data["comment"]["content"]
        discussion_title = webhook_data["discussion"]["title"]
        repo_name = webhook_data["repo"]["name"]

        # 从评论和讨论标题中提取潜在标签
        comment_tags = extract_tags_from_text(comment_content)
        title_tags = extract_tags_from_text(discussion_title)
        all_tags = list(set(comment_tags + title_tags))

        print(f"🔍 All unique tags: {all_tags}")

        if not all_tags:
            return ["No recognizable tags found in the discussion."]
    except Exception as e:                       # 修复：外层 try 缺 except，补上(解析 webhook 失败兜底)
        return [f"Error parsing webhook: {e}"]

    # 获取代理实例
    agent = await get_agent()
    if not agent:
        return ["Error: Agent not configured (missing HF_TOKEN)"]

    # 处理每个标签
    result_messages = []
    for tag in all_tags:
        try:
            # 使用代理处理标签
            prompt = f"""
            For the repository '{repo_name}', check if the tag '{tag}' already exists.
            If it doesn't exist, add it via a pull request.
    
            Repository: {repo_name}
            Tag to check/add: {tag}
            """

            print(f"🤖 Processing tag '{tag}' for repo '{repo_name}'")
            response = await agent.run(prompt)

            # 解析代理响应以确定成功/失败
            if "success" in response.lower():
                result_messages.append(f"✅ Tag '{tag}' processed successfully")
            else:
                result_messages.append(f"⚠️ Issue with tag '{tag}': {response}")

        except Exception as e:
            error_msg = f"❌ Error processing tag '{tag}': {str(e)}"
            print(error_msg)
            result_messages.append(error_msg)

    return result_messages



import re
from typing import List

# Recognized ML/AI tags for validation
RECOGNIZED_TAGS = {
    "pytorch", "tensorflow", "jax", "transformers", "diffusers",
    "text-generation", "text-classification", "question-answering",
    "text-to-image", "image-classification", "object-detection",
    "fill-mask", "token-classification", "translation", "summarization",
    "feature-extraction", "sentence-similarity", "zero-shot-classification",
    "image-to-text", "automatic-speech-recognition", "audio-classification",
    "voice-activity-detection", "depth-estimation", "image-segmentation",
    "video-classification", "reinforcement-learning", "tabular-classification",
    "tabular-regression", "time-series-forecasting", "graph-ml", "robotics",
    "computer-vision", "nlp", "cv", "multimodal",
}

def extract_tags_from_text(text: str) -> List[str]:
    """Extract potential tags from discussion text"""
    text_lower = text.lower()
    explicit_tags = []

    # 模式 1: "tag: something" 或 "tags: something"
    tag_pattern = r"tags?:\s*([a-zA-Z0-9-_,\s]+)"
    matches = re.findall(tag_pattern, text_lower)
    for match in matches:
        tags = [tag.strip() for tag in match.split(",")]
        explicit_tags.extend(tags)

    # 模式 2: "#hashtag" 样式
    hashtag_pattern = r"#([a-zA-Z0-9-_]+)"
    hashtag_matches = re.findall(hashtag_pattern, text_lower)
    explicit_tags.extend(hashtag_matches)

    # 模式 3: Look对于自然文本中提及的已识别标签，
    mentioned_tags = []
    for tag in RECOGNIZED_TAGS:
        if tag in text_lower:
            mentioned_tags.append(tag)

    # 合并并去重
    all_tags = list(set(explicit_tags + mentioned_tags))

    # 过滤，仅包含已识别标签或明确提及的标签
    valid_tags = []
    for tag in all_tags:
        if tag in RECOGNIZED_TAGS or tag in explicit_tags:
            valid_tags.append(tag)

    return valid_tags

#后台任务处理
from fastapi import FastAPI, BackgroundTasks

app = FastAPI(title="HF Tagging Bot")  # 片段示例先建一个 app；完整应用见下方"FastAPI Webhook 应用程序"


@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    """Handle webhook and process in background"""

    # 快速验证 webhook
    if request.headers.get("X-Webhook-Secret") != WEBHOOK_SECRET:
        return {"error": "Invalid secret"}

    webhook_data = await request.json()

    # 在后台运行以快速返回
    background_tasks.add_task(process_webhook_comment, webhook_data)

    return {"status": "accepted"}


#FastAPI Webhook 应用程序
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 配置
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
HF_TOKEN = os.getenv("HF_TOKEN")

# 用于存储已处理操作的简单存储
tag_operations_store: List[Dict[str, Any]] = []

app = FastAPI(title="HF Tagging Bot")
app.add_middleware(CORSMiddleware, allow_origins=["*"])



#. Webhook 数据模型

class  WebhookEvent ( BaseModel ):
    event: Dict [ str , str ]           # 包含操作和作用域信息
    comment: Dict [ str , Any ]         # 评论内容和元数据
    discussion: Dict [ str , Any ]      # 讨论信息
    repo: Dict [ str , str ]            # 仓库详情


@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    """
    Handle incoming webhooks from Hugging Face Hub
    Following the pattern from: https://raw.githubusercontent.com/huggingface/hub-docs/refs/heads/main/docs/hub/webhooks-guide-discussion-bot.md
    """
    print("🔔 Webhook received!")

    # 步骤 1：验证 webhook 密钥（安全）
    webhook_secret = request.headers.get("X-Webhook-Secret")
    if webhook_secret != WEBHOOK_SECRET:
        print("❌ Invalid webhook secret")
        return {"error": "incorrect secret"}, 400
    # 步骤 2：解析 webhook 数据
    try:
        webhook_data = await request.json()
        print(f"📥 Webhook data: {json.dumps(webhook_data, indent=2)}")
    except Exception as e:
        print(f"❌ Error parsing webhook data: {str(e)}")
        return {"error": "invalid JSON"}, 400

    # 步骤 3：验证事件结构
    event = webhook_data.get("event", {})
    if not event:
        print("❌ No event data in webhook")
        return {"error": "missing event data"}, 400
    # 第 4 步：检查这是否是讨论评论创建
    # 遵循 webhook 指南模式：
    if (
            event.get("action") == "create" and
            event.get("scope") == "discussion.comment"
    ):
        print("✅ Valid discussion comment creation event")

        # 在后台运行以快速返回 Hub
        background_tasks.add_task(process_webhook_comment, webhook_data)

        return {
            "status": "accepted",
            "message": "Comment processing started",
            "timestamp": datetime.now().isoformat()
        }
    else:
        print(f"ℹ️ Ignoring event: action={event.get('action')}, scope={event.get('scope')}")
        return {
            "status": "ignored",
            "reason": "Not a discussion comment creation"
        }


async def process_webhook_comment(webhook_data: Dict[str, Any]):
    """
    Process webhook comment to detect and add tags
    Integrates with our MCP client for Hub interactions
    """
    print("🏷️ Starting process_webhook_comment...")

    try:
        # 提取评论和仓库信息
        comment_content = webhook_data["comment"]["content"]
        discussion_title = webhook_data["discussion"]["title"]
        repo_name = webhook_data["repo"]["name"]
        discussion_num = webhook_data["discussion"]["num"]
        comment_author = webhook_data["comment"]["author"].get("id", "unknown")

        print(f"📝 Comment from {comment_author}: {comment_content}")
        print(f"📰 Discussion: {discussion_title}")
        print(f"📦 Repository: {repo_name}")
    except Exception as e:                    # 修复：补上缺失的 except
        return [f"Error parsing webhook: {e}"]

    # 从评论和标题中提取潜在标签
    comment_tags = extract_tags_from_text(comment_content)
    title_tags = extract_tags_from_text(discussion_title)
    all_tags = list(set(comment_tags + title_tags))

    print(f"🔍 Found tags: {all_tags}")

    # 商店操作监控
    operation = {
        "timestamp": datetime.now().isoformat(),
        "repo_name": repo_name,
        "discussion_num": discussion_num,
        "comment_author": comment_author,
        "extracted_tags": all_tags,
        "comment_preview": comment_content[:100] + "..." if len(comment_content) > 100 else comment_content,
        "status": "processing"
    }
    tag_operations_store.append(operation)
    if not all_tags:
        operation["status"] = "no_tags"
        operation["message"] = "No recognizable tags found"
        print("❌ No tags found to process")
        return

    # 获取用于标签处理的 MCP 代理 agent
    agent = await get_agent()
    if not agent:
        operation["status"] = "error"
        operation["message"] = "Agent not configured (missing HF_TOKEN)"
        print("❌ No agent available")
        return

    # 处理每个提取的标签
    operation["results"] = []
    for tag in all_tags:
        try:
            print(f"🤖 Processing tag '{tag}' for repo '{repo_name}'")

            # 创建提示，供代理处理标签处理
            prompt = f"""
            Analyze the repository '{repo_name}' and determine if the tag '{tag}' should be added.

            First, check the current tags using get_current_tags.
            If '{tag}' is not already present and it's a valid tag, add it using add_new_tag.

            Repository: {repo_name}
            Tag to process: {tag}

            Provide a clear summary of what was done.
            """

            response = await agent.run(prompt)
            print(f"🤖 Agent response for '{tag}': {response}")

            # 解析响应并存储结果
            tag_result = {
                "tag": tag,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            operation["results"].append(tag_result)

        except Exception as e:
            error_msg = f"❌ Error processing tag '{tag}': {str(e)}"
            print(error_msg)
            operation["results"].append({
                "tag": tag,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    operation["status"] = "completed"
    print(f"✅ Completed processing {len(all_tags)} tags")


#健康与监测终点
@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "name": "HF Tagging Bot",
        "status": "running",
        "description": "Webhook listener for automatic model tagging",
        "endpoints": {
            "webhook": "/webhook",
            "health": "/health",
            "operations": "/operations"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    agent = await get_agent()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "webhook_secret": "configured" if WEBHOOK_SECRET else "missing",
            "hf_token": "configured" if HF_TOKEN else "missing",
            "mcp_agent": "ready" if agent else "not_ready"
        }
    }
@app.get("/operations")
async def get_operations():
    """Get recent tag operations for monitoring"""
    # Return last 50 operations
    recent_ops = tag_operations_store[-50:] if tag_operations_store else []
    return {
        "total_operations": len(tag_operations_store),
        "recent_operations": recent_ops
    }

#开发仿真终点向 FastAPI 应用程序添加模拟端点
@app.post("/simulate_webhook")
async def simulate_webhook(
        repo_name: str,
        discussion_title: str,
        comment_content: str
) -> str:
    """Simulate webhook for testing purposes"""

    # 创建模拟 webhook 数据
    mock_webhook_data = {
        "event": {
            "action": "create",
            "scope": "discussion.comment"
        },
        "comment": {
            "content": comment_content,
            "author": {"id": "test-user"}
        },
        "discussion": {
            "title": discussion_title,
            "num": 999
        },
        "repo": {
            "name": repo_name
        }
    }

    # 处理模拟 webhook
    await process_webhook_comment(mock_webhook_data)

    return f"Simulated webhook processed for {repo_name}"
