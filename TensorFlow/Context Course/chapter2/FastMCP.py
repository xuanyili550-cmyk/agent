"""
================================================================================
 Context Course · Chapter 2 · 用 FastMCP 写 MCP 服务器（学习笔记描述）
================================================================================
 一句话：官方 FastMCP 几行就把普通函数暴露成 MCP 工具(自动生成 schema、跑 stdio/HTTP)。
 本章讲：
   ① FastMCP("name") + @mcp.tool() 装饰器登记工具(add/multiply)。
   ② 带参数/带类型注解的工具，docstring 即工具说明。
   ③ mcp.run() 起服务，供 Host(如 Claude Desktop)连接。
 要点：和手写 JSON-RPC 是同一套协议，FastMCP 帮你省样板。对照 实战练习/mcp实战/案例6。
 说明：需 mcp[cli]；可本机起服务。
================================================================================
"""

#pip install "mcp[cli]"
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

if __name__ == "__main__":
    mcp.run()

#更复杂的带参数工具

from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("file-analyzer")


@mcp.tool()
def read_file(path: str) -> str:
    """Read the contents of a file.

    Args:
        path: The absolute file path to read

    Returns:
        The file contents as a string
    """
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at {path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@mcp.tool()
def count_lines(path: str) -> int:
    """Count the number of lines in a file.

    Args:
        path: The absolute file path

    Returns:
        The number of lines in the file
    """
    try:
        with open(path, 'r') as f:
            return len(f.readlines())
    except Exception as e:
        return -1


@mcp.tool()
def list_directory(path: str) -> list[str]:
    """List files in a directory.

    Args:
        path: The directory path

    Returns:
        A list of file names in the directory
    """
    try:
        return os.listdir(path)
    except Exception as e:
        return []


if __name__ == "__main__":
    mcp.run()


#向服务器添加资源

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("documentation")

# Define resources
@mcp.resource("doc://api/overview")
def api_overview() -> str:
    """API overview and getting started guide."""
    return """# API Overview

This API provides tools for user management, data querying, and report generation.

## Getting Started
1. Authenticate with your API token
2. Call endpoints with appropriate parameters
3. Handle responses and errors gracefully

## Rate Limits
- 100 requests per minute per API key
- Burst limit: 10 requests per second
"""

@mcp.resource("doc://api/endpoints")
def api_endpoints() -> str:
    """Complete list of API endpoints."""
    return """# API Endpoints

## Users
- GET /users - List all users
- POST /users - Create a user
- GET /users/{id} - Get a specific user
- PUT /users/{id} - Update a user

## Data
- POST /query - Execute a database query
- GET /data/{id} - Retrieve data

## Reports
- GET /reports - List reports
- POST /reports - Generate a report
"""

@mcp.tool()
def get_api_status() -> dict:
    """Check the current API status."""
    return {
        "status": "operational",
        "uptime_percent": 99.99,
        "response_time_ms": 45
    }

if __name__ == "__main__":
    mcp.run()
#添加提示
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("prompts-server")

@mcp.prompt()
def code_review_prompt(language: str = "python") -> str:
    """Template for reviewing code in a specific language."""
    return f"""You are an expert {language} code reviewer. 
Analyze the provided code and provide feedback on:
1. Correctness and logic
2. Performance and efficiency
3. Code style and readability
4. Security implications
5. Testing coverage

Be constructive and suggest improvements."""

@mcp.prompt()
def security_audit_prompt() -> str:
    """Template for security audits."""
    return """Conduct a security audit of the provided code or system. Check for:
1. Authentication vulnerabilities
2. Authorization issues
3. Data validation gaps
4. SQL injection or injection attacks
5. Insecure dependencies
6. Sensitive information exposure

Rate each finding by severity (critical, high, medium, low)."""

@mcp.tool()
def get_available_prompts() -> list[str]:
    """List available prompt templates."""
    return ["code_review_prompt", "security_audit_prompt"]

if __name__ == "__main__":
    mcp.run()

#FastMCP中的错误处理
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("safe-operations")


@mcp.tool()
def divide(numerator: float, denominator: float) -> str:
    """Divide two numbers.

    Args:
        numerator: The dividend
        denominator: The divisor

    Returns:
        The quotient or an error message
    """
    try:
        if denominator == 0:
            return "Error: Cannot divide by zero"
        result = numerator / denominator
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def parse_json(data: str) -> str:
    """Parse a JSON string.

    Args:
        data: JSON string to parse

    Returns:
        Parsed JSON or error message
    """
    import json
    try:
        parsed = json.loads(data)
        return f"Valid JSON: {parsed}"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {str(e)}"


if __name__ == "__main__":
    mcp.run()

#Gradio MCP 集成

import gradio as gr

def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a letter in text.

    Args:
        word: The input text
        letter: The letter to search for

    Returns:
        The count of the letter
    """
    return word.lower().count(letter.lower())

def reverse_text(text: str) -> str:
    """Reverse a string.

    Args:
        text: The input text

    Returns:
        The reversed text
    """
    return text[::-1]

# Create UI
with gr.Blocks() as demo:
    with gr.Tab("Letter Counter"):
        word_input = gr.Textbox(label="Enter text")
        letter_input = gr.Textbox(label="Enter letter")
        count_output = gr.Number(label="Count")
        gr.Button("Count").click(letter_counter, [word_input, letter_input], count_output)

    with gr.Tab("Text Reversal"):
        text_input = gr.Textbox(label="Enter text")
        reversed_output = gr.Textbox(label="Reversed")
        gr.Button("Reverse").click(reverse_text, [text_input], reversed_output)

if __name__ == "__main__":
    demo.launch(mcp_server=True)

#构建一个实用的 Gradio MCP 应用
import gradio as gr
import json


def analyze_text(text: str) -> str:
    """Analyze text and compute statistics.

    Args:
        text: The input text to analyze

    Returns:
        JSON with analysis results
    """
    words = text.split()
    chars = len(text)

    return json.dumps({
        "words": len(words),
        "characters": chars,
        "average_word_length": round(chars / len(words), 2) if words else 0
    })


def reverse_text(text: str) -> str:
    """Reverse a string.

    Args:
        text: Input text

    Returns:
        The reversed text
    """
    return text[::-1]


def count_vowels(text: str) -> int:
    """Count vowels in text.

    Args:
        text: Input text

    Returns:
        Number of vowels
    """
    vowels = "aeiouAEIOU"
    return sum(1 for char in text if char in vowels)


# Create interface
with gr.Blocks(title="Text Tools") as demo:
    gr.Markdown("# Text Processing Tools")

    with gr.Tab("Analyze Text"):
        text_input1 = gr.Textbox(label="Enter text", lines=5)
        analysis_output = gr.Textbox(label="Analysis", lines=5)
        gr.Button("Analyze").click(analyze_text, text_input1, analysis_output)

    with gr.Tab("Reverse Text"):
        text_input2 = gr.Textbox(label="Enter text", lines=5)
        reverse_output = gr.Textbox(label="Reversed", lines=5)
        gr.Button("Reverse").click(reverse_text, text_input2, reverse_output)

    with gr.Tab("Count Vowels"):
        text_input3 = gr.Textbox(label="Enter text")
        vowel_output = gr.Number(label="Vowel Count")
        gr.Button("Count").click(count_vowels, text_input3, vowel_output)

if __name__ == "__main__":
    demo.launch(mcp_server=True)

#高级 Gradio MCP 功能
import gradio as gr


@gr.mcp.resource("config://api")
def api_documentation() -> str:
    """API documentation and endpoints."""
    return """# API Documentation

## Endpoints
- GET /users - List all users
- POST /users - Create user
- GET /users/{id} - Get specific user

## Authentication
Use Bearer token in Authorization header.
"""


# Your tools and UI here...

if __name__ == "__main__":
    demo.launch(mcp_server=True)
#仅限 MCP 的函数，使用 @gr.api()
import gradio as gr


@gr.api()
def query_database(sql: str) -> str:
    """Execute a database query (MCP only).

    Args:
        sql: SQL query string

    Returns:
        Query results
    """
    # Only accessible via MCP, not web UI
    return "Query results..."


@gr.api()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (MCP only).

    Args:
        to: Recipient email
        subject: Email subject
        body: Email body

    Returns:
        Confirmation message
    """
    # Only accessible via MCP
    return f"Email sent to {to}"


# Regular UI components here...

if __name__ == "__main__":
    demo.launch(mcp_server=True)


import gradio as gr

def authenticate(username: str, password: str) -> bool:
    return username == "admin" and password == "secret"

# `auth` protects the web UI only; MCP remains open.
if __name__ == "__main__":
    demo.launch(mcp_server=True, auth=authenticate)


# Good: Clear types, docstring, Args/Returns sections
def process_data(data: str, format: str = "json") -> str:
    """Process data in specified format.

    Args:
        data: Input data string
        format: Output format (default: json)

    Returns:
        Processed data as string
    """
    return data


# Good: Handles multiple parameters
def calculate(a: float, b: float, operation: str) -> float:
    """Perform mathematical operation.

    Args:
        a: First number
        b: Second number
        operation: Operation (add, subtract, multiply, divide)

    Returns:
        Result of operation
    """
    if operation == "add":
        return a + b
    # ... more operations


# Avoid: Missing type hints
def bad_function(data):  # No return type!
    return data


# Avoid: No docstring
def also_bad(text: str) -> str:
    return text.upper()


# Avoid: Complex types without clarification
def confusing(data: dict) -> list:  # What's in the dict/list?
    return []
#Gradio MCP故障排除
# 未显示为 MCP 工具的功能
# 检查你的函数签名：
#
# 所有参数都必须有类型提示
# 必须有文档字符串
# 文档字符串必须包含“Args:”部分
# 必须指定返回类型
# This will work
def my_tool(text: str) -> str:
    """Process text.

    Args:
        text: Input text

    Returns:
        Processed text
    """
    return text.upper()


# This won't work (no docstring)
def bad_tool(text: str) -> str:
    return text.upper()
# MCP 端点无响应
# 如果端点无法正常工作：
#
# 检查空间是否正在运行（未处于错误状态）
# 使用正确的端点：https://user-space.hf.space/gradio_api/mcp/
# 使用 curl 进行测试：curl https://user-space.hf.space/gradio_api/mcp/
# 检查空间日志是否存在错误
# 类型不匹配错误
# 如果返回类型与规范不符，代理可能会失败：
# Correct: Returns string as declared
def correct(x: int) -> str:
    return str(x * 2)

# Wrong: Declares string but returns int
def wrong(x: int) -> str:
    return x * 2  # Returns int, not string!
# 生产空间性能提升技巧
# 保持功能快速运行——代理程序会在大约 60 秒后超时。
# 验证输入— 处理前检查数据大小
# 优雅地处理错误——返回错误信息而不是导致程序崩溃。
# 缓存高成本结果——避免冗余计算
# 监控资源使用情况——空间CPU/内存有限
import time


def analyze(data: str) -> str:
    """Analyze large datasets.

    Args:
        data: Input data

    Returns:
        Analysis results
    """
    # Reject too-large inputs
    if len(data) > 1_000_000:
        return "Error: Input too large (max 1MB)"

    # Show progress
    start = time.time()
    # ... do analysis
    elapsed = time.time() - start

    if elapsed > 30:
        return "Error: Processing took too long"

    return "Results..."
#第一部分：在本地构建服务器
# mkdir text-processor-mcp
# cd text-processor-mcp
# python -m venv venv
# source venv/bin/activate  # On Windows: venv\Scripts\activate
# pip install "mcp[cli]"
from mcp.server.fastmcp import FastMCP
import json

mcp = FastMCP("text-processor")


@mcp.tool()
def analyze_text(text: str) -> str:
    """Analyze text and return statistics.

    Args:
        text: The input text to analyze

    Returns:
        JSON string with analysis results
    """
    words = text.split()
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    sentences = text.count(".") + text.count("!") + text.count("?")

    avg_word_length = round(chars_no_spaces / len(words), 2) if words else 0
    avg_sentence_length = round(len(words) / max(sentences, 1), 2)

    return json.dumps({
        "total_characters": chars,
        "characters_without_spaces": chars_no_spaces,
        "total_words": len(words),
        "total_sentences": max(sentences, 1),
        "average_word_length": avg_word_length,
        "average_sentence_length": avg_sentence_length,
        "unique_words": len(set(word.lower() for word in words))
    })


@mcp.tool()
def extract_keywords(text: str, count: int = 5) -> str:
    """Extract keywords (most common words) from text.

    Args:
        text: The input text
        count: Number of keywords to return (default 5)

    Returns:
        JSON string with keywords and frequencies
    """
    # Remove common words
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "is", "are", "was", "were", "be", "been", "by", "from"
    }

    words = text.lower().split()
    filtered = [w.strip(".,!?;:") for w in words if w.lower() not in stopwords]

    from collections import Counter
    word_freq = Counter(filtered)
    top_words = word_freq.most_common(count)

    return json.dumps({
        "keywords": [{"word": w, "frequency": f} for w, f in top_words]
    })


@mcp.tool()
def check_reading_level(text: str) -> str:
    """Estimate reading difficulty level.

    Args:
        text: The input text

    Returns:
        JSON string with reading level estimate
    """
    sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
    words = len(text.split())
    syllables = text.count("a") + text.count("e") + text.count("i") + text.count("o") + text.count("u")

    if words == 0:
        return json.dumps({"error": "No text to analyze"})

    # Flesch Kincaid Grade
    grade = (0.39 * (words / sentences)) + (11.8 * (syllables / words)) - 15.59
    grade = max(0, round(grade, 1))

    if grade < 6:
        level = "Elementary School"
    elif grade < 9:
        level = "Middle School"
    elif grade < 13:
        level = "High School"
    else:
        level = "College/Academic"

    return json.dumps({
        "grade_level": grade,
        "reading_level": level
    })


@mcp.tool()
def reverse_text(text: str) -> str:
    """Reverse a string.

    Args:
        text: The input text

    Returns:
        The reversed text
    """
    return text[::-1]


if __name__ == "__main__":
    mcp.run()

#第二部分：创建 Gradio Web UI + MCP 服务器
import gradio as gr
import json


def analyze_text(text: str) -> str:
    """Analyze text and return statistics.

    Args:
        text: The input text to analyze

    Returns:
        JSON string with analysis results
    """
    words = text.split()
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    sentences = text.count(".") + text.count("!") + text.count("?")

    avg_word_length = round(chars_no_spaces / len(words), 2) if words else 0
    avg_sentence_length = round(len(words) / max(sentences, 1), 2)

    return json.dumps({
        "total_characters": chars,
        "characters_without_spaces": chars_no_spaces,
        "total_words": len(words),
        "total_sentences": max(sentences, 1),
        "average_word_length": avg_word_length,
        "average_sentence_length": avg_sentence_length
    }, indent=2)


def extract_keywords(text: str, count: int = 5) -> str:
    """Extract keywords (most common words) from text.

    Args:
        text: The input text
        count: Number of keywords to return (default 5)

    Returns:
        JSON string with keywords and frequencies
    """
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "is", "are", "was", "were", "be", "been", "by", "from"
    }

    words = text.lower().split()
    filtered = [w.strip(".,!?;:") for w in words if w.lower() not in stopwords]

    from collections import Counter
    word_freq = Counter(filtered)
    top_words = word_freq.most_common(count)

    return json.dumps({
        "keywords": [{"word": w, "frequency": f} for w, f in top_words]
    }, indent=2)


def check_reading_level(text: str) -> str:
    """Estimate reading difficulty level.

    Args:
        text: The input text

    Returns:
        JSON string with reading level estimate
    """
    sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
    words = len(text.split())
    vowels = "aeiou"
    syllables = sum(1 for c in text.lower() if c in vowels)

    if words == 0:
        return json.dumps({"error": "No text to analyze"})

    grade = max(0, (0.39 * (words / sentences)) + (11.8 * (syllables / words)) - 15.59)

    if grade < 6:
        level = "Elementary School"
    elif grade < 9:
        level = "Middle School"
    elif grade < 13:
        level = "High School"
    else:
        level = "College/Academic"

    return json.dumps({
        "grade_level": round(grade, 1),
        "reading_level": level
    }, indent=2)


# Create web UI
with gr.Blocks(title="Text Processor") as demo:
    gr.Markdown("# Text Processing Tools")
    gr.Markdown("Analyze text statistics, extract keywords, and check reading difficulty.")

    with gr.Tab("Analyze Text"):
        text_input1 = gr.Textbox(
            label="Enter text",
            lines=8,
            placeholder="Paste your text here..."
        )
        analysis_output = gr.Textbox(label="Analysis Results", lines=8)
        gr.Button("Analyze", size="lg").click(analyze_text, text_input1, analysis_output)

    with gr.Tab("Extract Keywords"):
        text_input2 = gr.Textbox(label="Enter text", lines=8)
        count_input = gr.Slider(1, 20, value=5, step=1, label="Number of keywords")
        keywords_output = gr.Textbox(label="Keywords", lines=8)
        gr.Button("Extract", size="lg").click(
            extract_keywords,
            [text_input2, count_input],
            keywords_output
        )

    with gr.Tab("Reading Level"):
        text_input3 = gr.Textbox(label="Enter text", lines=8)
        level_output = gr.Textbox(label="Reading Level Analysis", lines=5)
        gr.Button("Check Level", size="lg").click(check_reading_level, text_input3, level_output)

if __name__ == "__main__":
    demo.launch(mcp_server=True)

#挑战扩展
# 尝试以下改进措施：
#
# 添加情感分析——检测积极/消极语气
# 添加语言检测功能——识别文本语言
# 添加文本摘要— 创建简短摘要
# 添加拼写检查功能——识别拼写错误的单词
# 添加可读性提示——提出改进建议以提高清晰度