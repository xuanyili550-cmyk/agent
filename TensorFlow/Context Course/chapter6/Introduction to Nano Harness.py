"""
================================================================================
 Context Course · Chapter 6 · Nano Harness：读懂 agent 循环底层（学习笔记描述）
================================================================================
 一句话：~220 行的极简 agent，把系统提示/步循环/工具执行/消息历史/错误处理/沙箱摊开给你读。
 本章讲：
   ① 主流框架(Claude Code/Codex)把循环藏起来，学习就得看最小实现。
   ② Nano Harness 的组成：接任务→步循环→调工具→更新历史→错误重试→沙箱边界。
   ③ 用单一模型(GLM 等)让"唯一变量"只剩循环和工具。
 要点：从零读最小 harness 最能看清 agent"到底怎么转"。对照 实战练习/生产工程补齐/补3(手写ReAct)。
 说明：学习用(非生产)，参考为主。
================================================================================
"""

# # Introduction to Nano Harness
#
# Mainstream agent frameworks (Codex, Claude Code, OpenCode) hide the loop that makes them work, which is great for production use, but not for learning.
#
# Nano Harness is a ~220-line Python agent built for reading, not production. It shows the system prompt, the step loop, tool execution, message history, error handling, and sandboxing in one file.
#
# > [!WARNING]
# > nano_harness is a learning tool and not intended for production use. It's just a fun way to understand how agents work under the hood. Do not use it for real work!
#
# ## Why start from scratch?
#
# Learning from scratch is a great way to understand how anything works under the hood.
#
# Reading a minimal harness makes the design choices legible: when to retry, how to parse output, how to handle errors, and where the security boundary sits. This unit uses `zai-org/GLM-5.1` via Hugging Face Inference Providers as the single model, so the only moving parts are the loop and the tools.
#
# ## What is Nano Harness?
#
# Nano Harness is a ~220-line Python agent framework that:
#
# 1. **Takes a task** — "Inspect the workspace and provide a summary"
# 2. **Calls an LLM** — Uses OpenAI-compatible API (defaults to HF router)
# 3. **Generates Python code** — Model outputs executable Python code that will help complete the task
# 4. **Executes safely** — In a sandboxed environment with allowed tools
# 5. **Observes results** — stdout, stderr, exceptions, or a captured final answer
# 6. **Updates context** — Feeds observations back to the model
# 7. **Repeats** — Until the task is done or we hit step limit (we're assuming 50 steps in our implementation)
#
# ## Key Features
#
# ### Code-First Agent
# Instead of returning JSON or text, the model outputs Python code:
#
# ```python
# # Model says:
# files = list_dir(".")
# models = read_file("models.txt")
# final_answer("Files:\n" + "\n".join(files) + "\n\nmodels.txt:\n" + models)
# ```
#
# The agent parses and executes it.
#
# ### Constrained Tools
# Four tools available:
#
# | Tool | What It Does | Security |
# |------|-------------|----------|
# | `list_dir(path)` | List directory contents | Path confinement |
# | `read_file(path, max_chars)` | Read file | Path confinement, size limit |
# | `write_file(path, content)` | Create/modify file | Path confinement, disabled by default |
# | `exec_cmd(args)` | Run shell command | Allowlist (only: ls, cat, pwd, echo, head, tail, wc, rg) |
#
# ### Sandboxed Execution
# - All file access confined to workspace
# - Commands restricted to allowlist
# - Output size limited to prevent context overflow
# - Timeouts prevent hanging
#
# ### Model via Hugging Face Inference Providers
#
# The harness defaults to the Hugging Face router, which exposes an OpenAI-compatible `/v1` surface backed by Inference Providers:
#
# ```bash
# export NANO_MODEL="zai-org/GLM-5.1"
# export HF_TOKEN="hf_..."
# ```
#
# ## The Agentic Loop
#
# Nano Harness implements the core loop:
#
# ```
# 1. Call LLM with task + message history
# 2. Parse model output as Python code
# 3. Execute Python (with tools available)
# 4. Collect stdout, stderr, exceptions
# 5. Append observation to message history
# 6. Repeat (max 50 steps)
# 7. Done when: final_answer() called or max steps reached
# ```
#
# ## What You'll Learn
#
# This unit walks through the agent loop code piece by piece, explains how the tools are designed and sandboxed, extends the harness with `web_fetch` and HF Hub search, and runs it against `zai-org/GLM-5.1` through Hugging Face Inference Providers.
#
# ## Prerequisites
#
# Basic Python, familiarity with HTTP APIs and sandboxing, and an HF token with access to Inference Providers.
#
# # Introduction to Nano Harness
#
# Mainstream agent frameworks (Codex, Claude Code, OpenCode) hide the loop that makes them work, which is great for production use, but not for learning.
#
# Nano Harness is a ~220-line Python agent built for reading, not production. It shows the system prompt, the step loop, tool execution, message history, error handling, and sandboxing in one file.
#
# > [!WARNING]
# > nano_harness is a learning tool and not intended for production use. It's just a fun way to understand how agents work under the hood. Do not use it for real work!
#
# ## Why start from scratch?
#
# Learning from scratch is a great way to understand how anything works under the hood.
#
# Reading a minimal harness makes the design choices legible: when to retry, how to parse output, how to handle errors, and where the security boundary sits. This unit uses `zai-org/GLM-5.1` via Hugging Face Inference Providers as the single model, so the only moving parts are the loop and the tools.
#
# ## What is Nano Harness?
#
# Nano Harness is a ~220-line Python agent framework that:
#
# 1. **Takes a task** — "Inspect the workspace and provide a summary"
# 2. **Calls an LLM** — Uses OpenAI-compatible API (defaults to HF router)
# 3. **Generates Python code** — Model outputs executable Python code that will help complete the task
# 4. **Executes safely** — In a sandboxed environment with allowed tools
# 5. **Observes results** — stdout, stderr, exceptions, or a captured final answer
# 6. **Updates context** — Feeds observations back to the model
# 7. **Repeats** — Until the task is done or we hit step limit (we're assuming 50 steps in our implementation)
#
# ## Key Features
#
# ### Code-First Agent
# Instead of returning JSON or text, the model outputs Python code:
#
# ```python
# # Model says:
# files = list_dir(".")
# models = read_file("models.txt")
# final_answer("Files:\n" + "\n".join(files) + "\n\nmodels.txt:\n" + models)
# ```
#
# The agent parses and executes it.
#
# ### Constrained Tools
# Four tools available:
#
# | Tool | What It Does | Security |
# |------|-------------|----------|
# | `list_dir(path)` | List directory contents | Path confinement |
# | `read_file(path, max_chars)` | Read file | Path confinement, size limit |
# | `write_file(path, content)` | Create/modify file | Path confinement, disabled by default |
# | `exec_cmd(args)` | Run shell command | Allowlist (only: ls, cat, pwd, echo, head, tail, wc, rg) |
#
# ### Sandboxed Execution
# - All file access confined to workspace
# - Commands restricted to allowlist
# - Output size limited to prevent context overflow
# - Timeouts prevent hanging
#
# ### Model via Hugging Face Inference Providers
#
# The harness defaults to the Hugging Face router, which exposes an OpenAI-compatible `/v1` surface backed by Inference Providers:
#
# ```bash
# export NANO_MODEL="zai-org/GLM-5.1"
# export HF_TOKEN="hf_..."
# ```
#
# ## The Agentic Loop
#
# Nano Harness implements the core loop:
#
# ```
# 1. Call LLM with task + message history
# 2. Parse model output as Python code
# 3. Execute Python (with tools available)
# 4. Collect stdout, stderr, exceptions
# 5. Append observation to message history
# 6. Repeat (max 50 steps)
# 7. Done when: final_answer() called or max steps reached
# ```
#
# ## What You'll Learn
#
# This unit walks through the agent loop code piece by piece, explains how the tools are designed and sandboxed, extends the harness with `web_fetch` and HF Hub search, and runs it against `zai-org/GLM-5.1` through Hugging Face Inference Providers.
#
# ## Prerequisites
#
# Basic Python, familiarity with HTTP APIs and sandboxing, and an HF token with access to Inference Providers.
#
def list_dir(path="."):
    """List directory contents."""
    p = safe_path(path)  # Ensure path is in workspace
    if not p.is_dir():
        raise NotADirectoryError(str(p))
    return sorted([x.name + ("/" if x.is_dir() else "") for x in p.iterdir()])

def read_file(path, max_chars=4000):
    """Read file with size limit."""
    p = safe_path(path)
    content = p.read_text(encoding="utf-8", errors="replace")
    return clip(content, min(max_chars, MAX_CHARS))  # Limit output

def write_file(path, content):
    """Write or create file if writes are enabled."""
    if not ALLOW_WRITE:
        raise PermissionError("write_file disabled")
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(content), encoding="utf-8")
    return f"Wrote {len(str(content))} bytes"

def exec_cmd(args):
    """Execute shell command (whitelisted only)."""
    if args[0] not in ALLOW_COMMANDS:
        raise PermissionError(f"Command {args[0]} not allowed")
    result = subprocess.run(args, capture_output=True, timeout=TIMEOUT_S, text=True)
    output_parts = []
    if result.stdout:
        output_parts.append(f"stdout:\n{result.stdout}")
    if result.stderr:
        output_parts.append(f"stderr:\n{result.stderr}")
    output = "\n\n".join(output_parts) or f"(exit code {result.returncode} with no output)"
    return clip(output, MAX_CHARS)

DONE = False
FINAL_RESULT = None

def final_answer(value):
    """Agent calls this when task is complete."""
    global DONE, FINAL_RESULT
    DONE = True
    FINAL_RESULT = value
    return value


def main():
    global DONE, FINAL_RESULT
    DONE = False
    FINAL_RESULT = None
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TASK}
    ]

    for step in range(MAX_STEPS):
        print(f"\n[Step {step + 1}]")

        # 1. Call LLM
        response = client.responses.create(
            model=MODEL,
            temperature=TEMPERATURE,
            input=messages
        )

        content = response.output_text
        print(f"Model output:\n{content[:500]}...")

        # 2. Add model response to history
        messages.append({"role": "assistant", "content": content})

        # 3. Parse and execute Python code
        code = extract_python(content)  # Parse code block from response
        try:
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            exec_globals = {
                "__builtins__": {},
                "list_dir": list_dir,
                "read_file": read_file,
                "write_file": write_file,
                "exec_cmd": exec_cmd,
                "final_answer": final_answer
            }
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, exec_globals)

            stdout_text = stdout_buffer.getvalue().strip()
            stderr_text = stderr_buffer.getvalue().strip()

            if DONE:
                result = f"Final answer: {clip(FINAL_RESULT)}"
            else:
                observations = []
                if stdout_text:
                    observations.append(f"stdout:\n{clip(stdout_text)}")
                if stderr_text:
                    observations.append(f"stderr:\n{clip(stderr_text)}")
                result = "\n\n".join(observations) or "Executed successfully (no output)"
        except FileNotFoundError:
            result = "Error: FileNotFoundError: File not found"
        except PermissionError as e:
            result = f"Error: PermissionError: {str(e)}"
        except subprocess.TimeoutExpired:
            result = "Error: TimeoutError: Command took too long"
        except Exception as e:
            result = f"Error: {type(e).__name__}: {str(e)}"

        # 4. Check if agent called final_answer()
        if DONE:
            print(f"✓ Task complete: {FINAL_RESULT}")
            break

        # 5. Add observation to message history
        messages.append({"role": "user", "content": result})

    if not DONE:
        print(f"✗ Max steps ({MAX_STEPS}) reached without final_answer()")

#步限安全

for step in range(MAX_STEPS):  # MAX_STEPS = 50
    # ... run loop ...
    if DONE:
        break

if not DONE:
    print("Max steps reached without final_answer()")

#上下文管理

# Good: Agent reads one file at a time
read_file("test.py", max_chars=2000)  # 2000 chars ✓

# Bad: Agent tries to read entire codebase at once
read_file("large_codebase.py", max_chars=50000)  # Gets clipped to 8000

#工具和沙盒详解
def list_dir(path="."):
    """List files and directories."""
    p = safe_path(path)  # Verify path is in workspace
    if not p.is_dir():
        raise NotADirectoryError(str(p))
    return sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
list_dir(".")             # ✓ OK
list_dir("src")           # ✓ OK
list_dir("../etc")        # ✗ BLOCKED by safe_path()

#2. read_file(path, max_chars) — 读取文件

def read_file(path, max_chars=4000):
    """Read file with size limit."""
    p = safe_path(path)
    content = p.read_text(encoding="utf-8", errors="replace")
    # Enforce framework-level limit
    return clip(content, min(max_chars, MAX_CHARS))  # MAX_CHARS=8000

read_file("README.md")             # ✓ Up to 4000 chars
read_file("data.txt", max_chars=500)  # ✓ Up to 500 chars
read_file("huge.db")               # ✓ Clipped to 8000 chars (not unlimited)
read_file("/etc/passwd")           # ✗ BLOCKED by safe_path()
#write_file(path, content) — 写入文件
ALLOW_WRITE = False  # Disabled by default!


def write_file(path, content):
    """Write file (gated by ALLOW_WRITE flag)."""
    if not ALLOW_WRITE:
        raise PermissionError("write_file disabled")

    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(content), encoding="utf-8")
    return f"Wrote {len(str(content))} bytes to {p}"
# Enable write explicitly
ALLOW_WRITE = True

write_file("output.txt", "Hello")  # ✓ OK if ALLOW_WRITE=True
ALLOW_COMMANDS = ["ls", "cat", "pwd", "echo", "head", "tail", "wc", "rg"]


def exec_cmd(args):
    """Execute command (whitelist only)."""
    if args[0] not in ALLOW_COMMANDS:
        raise PermissionError(f"Command '{args[0]}' not allowed")

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            timeout=TIMEOUT_S,  # e.g., 30 seconds
            text=True
        )
        output_parts = []
        if result.stdout:
            output_parts.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"stderr:\n{result.stderr}")
        output = "\n\n".join(output_parts) or f"(exit code {result.returncode} with no output)"
        return clip(output, MAX_CHARS)
    except subprocess.TimeoutExpired:
        return "Error: Command timed out"

exec_cmd(["ls", "-la"])      # ✓ OK (ls whitelisted)
exec_cmd(["pwd"])            # ✓ OK
exec_cmd(["rg", "ERROR"])    # ✓ OK (rg is ripgrep, safe)
exec_cmd(["rm", "-rf", "/"]) # ✗ BLOCKED (rm not whitelisted)
exec_cmd(["curl", "evil.com"])  # ✗ BLOCKED (curl not whitelisted)

WORKSPACE = Path.cwd()  # e.g., /home/user/project


def safe_path(user_input):
    """Ensure path is within workspace."""
    # Resolve to absolute path
    requested = (WORKSPACE / user_input).resolve()

    # Check if it's inside workspace
    if not requested.is_relative_to(WORKSPACE):
        raise ValueError(f"Path {user_input} escapes workspace")

    return requested

# Attacker tries directory traversal
safe_path("../../../etc/passwd")
# → Resolves to /home/user/project/../../etc/passwd
# → Resolves to /etc/passwd
# → NOT relative to WORKSPACE
# → Raises ValueError ✓

# Attacker tries absolute path
safe_path("/etc/passwd")
# → Doesn't start with WORKSPACE
# → Raises ValueError ✓

# Legitimate use
safe_path("data/models.txt")
# → Resolves to /home/user/project/data/models.txt
# → IS relative to WORKSPACE
# → Returns path ✓

# Only these functions are available
exec_globals = {
    "__builtins__": {}, # need to explicitly remove builtins
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "exec_cmd": exec_cmd,
    "final_answer": final_answer
}

# Agent code runs here
exec(agent_code, exec_globals)

# Agent CANNOT do:
# - import modules (no __builtins__)
# - access files outside tools
# - use network directly
# - access parent process variables

try:
    exec(agent_code, exec_globals)
except Exception as e:
    error_msg = f"Error: {type(e).__name__}: {str(e)}"
    messages.append({"role": "user", "content": error_msg})
    # Continue to next iteration; agent adapts

# Step 1: Agent tries to read huge file
#   Code: content = read_file("huge.dat", max_chars=999999)
#   Result: (Clipped to 8000 chars, agent sees it)
#
# Step 2: Agent tries forbidden command
#   Code: exec_cmd(["rm", "data.txt"])
#   Result: PermissionError: Command 'rm' not allowed
#   Agent sees error and tries different approach
#
# Step 3: Agent tries to escape workspace
#   Code: read_file("../../../etc/passwd")
#   Result: ValueError: Path escapes workspace
#   Agent learns and adjusts

#设计自定义工具
def good_tool(user_input, max_result_size=1000):
    """
    1. Validate and confine input
    2. Perform operation
    3. Limit output size
    4. Return safe result
    """
    # 1. Validate input
    if not isinstance(user_input, str):
        raise TypeError("user_input must be string")

    path = safe_path(user_input)  # Confine paths

    # 2. Perform operation
    result = path.read_text()

    # 3. Limit output
    return clip(result, min(max_result_size, MAX_CHARS))
# ✗ BAD: No input validation
def bad_tool_1(path):
    return open(path).read()  # Reads anything!

# ✗ BAD: No output limit
def bad_tool_2(query):
    return database.query(query)  # Could be terabytes

# ✗ BAD: No error handling
def bad_tool_3(url):
    return requests.get(url).text  # Can timeout, hang

# ✗ BAD: Trusts agent completely
def bad_tool_4(command):
    os.system(command)  # Agent can run rm -rf /

#实践操作：扩展纳米线束
import urllib.request
import urllib.error

def web_fetch(url, max_bytes=10000):
    """Fetch web page content with size limit."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
            content = response.read(max_bytes + 1)
            if len(content) > max_bytes:
                content = content[:max_bytes] + b"\n...[truncated]"
            return content.decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return f"Error: Failed to fetch {url}: {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
import urllib.request
import urllib.error
SYSTEM_PROMPT = """
...
Tools:
  - web_fetch(url, max_bytes=10000) → fetch webpage
...
"""

# Agent writes:
content = web_fetch("https://huggingface.co/")

# Or with size limit:
content = web_fetch("https://huggingface.co/", max_bytes=5000)

#扩展 2：添加 hf_search 工具
def hf_search(query, resource_type="models", limit=5):
    """Search Hugging Face Hub (requires HF_TOKEN)."""
    if not API_KEY:
        return "Error: HF_TOKEN not set. Can't access Hugging Face API."

    try:
        url = f"https://huggingface.co/api/{resource_type}"
        params = f"?search={query}&limit={limit}"

        req = urllib.request.Request(
            url + params,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )

        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as response:
            data = json.loads(response.read())

            # Format results
            results = []
            for item in data[:limit]:
                results.append({
                    "id": item.get("id"),
                    "downloads": item.get("downloads", 0),
                    "description": item.get("description", "")[:200]
                })

            return results

    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
SYSTEM_PROMPT = """ 
...
工具：
  - hf_search(query, resource_type='models', limit=10) → 搜索 HF 
... 
"""

# Agent writes:
results = hf_search("bert", resource_type="models", limit=5)
final_answer(results)

#完整扩展示例
# !/usr/bin/env python3
import io
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from openai import OpenAI

# Configuration
TASK = "Search for bert models on Hugging Face and summarize top 3."
MODEL = os.getenv("NANO_MODEL", "zai-org/GLM-5.1")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
WORKSPACE = str(Path.cwd())
MAX_STEPS = 50
TIMEOUT_S = 30
MAX_CHARS = 8000
ALLOW_WRITE = False
ALLOW_COMMANDS = ["ls", "cat", "pwd", "echo", "head", "tail", "wc", "rg"]
TEMPERATURE = 0.2

SYSTEM_PROMPT = f"""You are a code-first agent.
Reply with executable Python only.

Tools:
  - list_dir(path='.') → list files
  - read_file(path, max_chars=4000) → read file
  - write_file(path, content) → write file (only if ALLOW_WRITE=True)
  - exec_cmd(args) → run allowed command
  - web_fetch(url, max_bytes=10000) → fetch webpage
  - hf_search(query, limit=5) → search HF Hub

Allowed commands: {ALLOW_COMMANDS}
Writes enabled: {ALLOW_WRITE}

When done, call final_answer(result).
Output only Python code, no prose."""


def clip(x, n=MAX_CHARS):
    s = str(x)
    return s[:n] + f"\n...[truncated]" if len(s) > n else s


def main():
    ws = Path(WORKSPACE).resolve()
    done = False
    final_result = None

    def safe_path(path):
        p = (ws / path).resolve()
        try:
            p.relative_to(ws)
        except ValueError:
            raise ValueError(f"Path escapes workspace: {path}")
        return p

    def list_dir(path="."):
        p = safe_path(path)
        return sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())

    def read_file(path, max_chars=4000):
        p = safe_path(path)
        return clip(p.read_text(errors="replace"), min(max_chars, MAX_CHARS))

    def write_file(path, content):
        if not ALLOW_WRITE:
            raise PermissionError("write_file disabled")
        p = safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(content), encoding="utf-8")
        return f"Wrote {len(str(content))} bytes"

    def exec_cmd(args):
        if args[0] not in ALLOW_COMMANDS:
            raise PermissionError(f"Command {args[0]} not allowed")
        result = subprocess.run(
            args, capture_output=True, timeout=TIMEOUT_S, text=True
        )
        output_parts = []
        if result.stdout:
            output_parts.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"stderr:\n{result.stderr}")
        output = "\n\n".join(output_parts) or f"(exit code {result.returncode} with no output)"
        return clip(output, MAX_CHARS)

    def web_fetch(url, max_bytes=10000):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
                content = r.read(max_bytes)
                return content.decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error: {type(e).__name__}: {str(e)}"

    def hf_search(query, resource_type="models", limit=5):
        if not API_KEY:
            return "Error: HF_TOKEN not set"
        try:
            url = f"https://huggingface.co/api/{resource_type}"
            req = urllib.request.Request(
                f"{url}?search={query}&limit={limit}",
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                data = json.loads(r.read())
                return [
                    {
                        "id": item.get("id"),
                        "downloads": item.get("downloads", 0),
                        "description": item.get("description", "")[:100]
                    }
                    for item in data[:limit]
                ]
        except Exception as e:
            return f"Error: {str(e)}"

    def final_answer(value):
        nonlocal done, final_result
        done = True
        final_result = value
        return value

    # Initialize
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TASK}
    ]

    # Main loop
    for step in range(MAX_STEPS):
        print(f"\n[Step {step + 1}]")

        # Call LLM
        response = client.responses.create(
            model=MODEL,
            temperature=TEMPERATURE,
            input=messages
        )

        content = response.output_text
        print(f"Model:\n{content[:300]}...")

        messages.append({"role": "assistant", "content": content})

        # Execute code
        try:
            code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
            if not code_match:
                raise ValueError("No Python code block found")

            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            exec_globals = {
                "__builtins__": {},
                "list_dir": list_dir,
                "read_file": read_file,
                "write_file": write_file,
                "exec_cmd": exec_cmd,
                "web_fetch": web_fetch,
                "hf_search": hf_search,
                "final_answer": final_answer,
                "json": json
            }

            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code_match.group(1), exec_globals)

            stdout_text = stdout_buffer.getvalue().strip()
            stderr_text = stderr_buffer.getvalue().strip()

            if done:
                result = f"Final answer: {clip(final_result)}"
            else:
                observations = []
                if stdout_text:
                    observations.append(f"stdout:\n{clip(stdout_text)}")
                if stderr_text:
                    observations.append(f"stderr:\n{clip(stderr_text)}")
                result = "\n\n".join(observations) or "Executed successfully (no output)"
        except FileNotFoundError:
            result = "Error: FileNotFoundError: File not found"
        except PermissionError as e:
            result = f"Error: PermissionError: {str(e)}"
        except subprocess.TimeoutExpired:
            result = "Error: TimeoutError: Command took too long"
        except Exception as e:
            result = f"Error: {type(e).__name__}: {str(e)}"

        if done:
            print(f"✓ Task complete: {final_result}")
            break

        messages.append({"role": "user", "content": result})

    if not done:
        print(f"✗ Max steps reached")


if __name__ == "__main__":
    main()

#进一步拓展
# Add "git" to ALLOW_COMMANDS first, then:
def git_log(limit=10):
    """Get recent git commits."""
    return exec_cmd(["git", "log", "--oneline", f"-{limit}"])
def json_parse(json_string):
    """Parse JSON safely."""
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        return f"Error: {str(e)}"
def compute_stats(numbers):
    """Compute min, max, mean."""
    nums = list(map(float, numbers))
    return {
        "min": min(nums),
        "max": max(nums),
        "mean": sum(nums) / len(nums),
        "count": len(nums)
    }