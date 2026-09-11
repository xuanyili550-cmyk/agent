"""
================================================================================
 MCP Course · Chapter 1 · Model Context Protocol 入门（可运行·看清协议长什么样）
================================================================================
 MCP(模型上下文协议) = Anthropic 提出的“AI 应用 ⇄ 外部能力”的开放标准，被称为
 “AI 界的 USB-C”：一次实现，处处可插。让 LLM 能统一地用工具、读数据、套模板。

 为什么需要 MCP：以前每个 AI 应用要和每个数据源/工具做私有对接(M×N 个集成)。
 MCP 定义统一协议后，服务器实现一次，任何兼容 MCP 的 Host(Claude Desktop/IDE/…)都能用(M+N)。

 三个角色：
   Host    用户在用的 AI 应用(Claude Desktop、Cursor…)，内含一个或多个 Client。
   Client  Host 里负责连某个 Server 的连接器(一对一)。
   Server  你写的程序，对外暴露 工具/资源/提示。

 四大原语(能力)：
   ┌────────┬─────────────────────────────┬──────────────────────────┐
   │ 工具    │ 可执行函数(AI 主动调用，会产生副作用/计算) │ get_weather(city)         │
   │ 资源    │ 只读数据源(提供上下文，几乎不计算)          │ 文件内容/数据库记录        │
   │ 提示    │ 预定义模板/工作流(指导交互)               │ 代码审查模板              │
   │ 采样    │ 服务器反向请求客户端跑 LLM(递归自我改进)    │ 写作 App 自审再改稿       │
   └────────┴─────────────────────────────┴──────────────────────────┘

 通信：JSON-RPC 2.0(消息格式) + 传输层(stdio 本地 / HTTP+SSE 远程)。
 动态发现：客户端连上后用 tools/list、resources/list、prompts/list 问服务器“你有啥”。

 直接运行：python3 chapter1_MCP协议入门_学习笔记.py
   本文件把“真实 MCP 消息”打印出来给你看；能真跑的服务器/客户端见 chapter2/3/4。
================================================================================
"""

import json


def show(title, obj):
    print(f"\n── {title} ──")
    print(json.dumps(obj, indent=2, ensure_ascii=False))


print("=" * 72 + "\n MCP 通信 = JSON-RPC 2.0 消息。下面是每一步真实长什么样：\n" + "=" * 72)

# ------------------------------------------------------------------------------
# 1) 握手 initialize：客户端和服务器交换协议版本与能力(capabilities)
# ------------------------------------------------------------------------------
show("① 客户端 → 服务器：initialize 请求", {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
               "clientInfo": {"name": "claude-desktop", "version": "1.0"}},
})
show("① 服务器 → 客户端：initialize 响应(声明我支持 tools/resources/prompts)", {
    "jsonrpc": "2.0", "id": 1,
    "result": {"protocolVersion": "2024-11-05",
               "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
               "serverInfo": {"name": "my-server", "version": "1.0.0"}},
})

# ------------------------------------------------------------------------------
# 2) 动态发现 tools/list：客户端问“你有哪些工具”
# ------------------------------------------------------------------------------
show("② 客户端 → 服务器：tools/list 请求", {
    "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
})
show("② 服务器 → 客户端：tools/list 响应(每个工具带 inputSchema 供 LLM 理解怎么调)", {
    "jsonrpc": "2.0", "id": 2,
    "result": {"tools": [{
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "inputSchema": {"type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]},
    }]},
})

# ------------------------------------------------------------------------------
# 3) 调用 tools/call：AI 决定调用工具
# ------------------------------------------------------------------------------
show("③ 客户端 → 服务器：tools/call 请求(带 name + arguments)", {
    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
    "params": {"name": "get_weather", "arguments": {"location": "Beijing"}},
})
show("③ 服务器 → 客户端：tools/call 响应(content 列表 + isError)", {
    "jsonrpc": "2.0", "id": 3,
    "result": {"content": [{"type": "text",
                            "text": '{"temperature": 22, "conditions": "Cloudy"}'}],
               "isError": False},
})

# ------------------------------------------------------------------------------
# 4) 资源 / 提示 的发现同理(resources/list、prompts/list)
# ------------------------------------------------------------------------------
show("④ 资源发现 resources/list 响应(资源用 URI 标识)", {
    "jsonrpc": "2.0", "id": 4,
    "result": {"resources": [{"uri": "file:///readme.md", "name": "README",
                              "mimeType": "text/markdown"}]},
})
show("④ 提示发现 prompts/list 响应(提示带参数定义)", {
    "jsonrpc": "2.0", "id": 5,
    "result": {"prompts": [{"name": "code_review",
                            "description": "Generate a code review.",
                            "arguments": [{"name": "code", "required": True}]}]},
})

print("""
================================================================================
 要点总结：
  · MCP = 统一协议，让“AI 应用”和“工具/数据”解耦(M×N → M+N)。
  · 角色：Host(应用) → Client(连接器) → Server(你暴露能力)。
  · 四原语：工具(会执行) / 资源(只读) / 提示(模板) / 采样(反向调 LLM)。
  · 消息：JSON-RPC 2.0；每次都有 method + params/result；靠 *_list 动态发现。
  · 传输：本地 stdio / 远程 HTTP+SSE，业务消息一模一样。
 下一步：chapter2 手写真服务器(tools) → chapter3 补齐(资源/提示) → chapter4 Agent 循环+官方SDK。
================================================================================""")
