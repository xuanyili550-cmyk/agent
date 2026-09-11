"""
================================================================================
 MCP 实战导航 · 用 MCP 把 HF 课程 Ch1-12 的能力暴露成"工具/资源/提示"
================================================================================
 主题：MCP(Model Context Protocol) 是 Anthropic 开源的开放标准，用统一的 JSON-RPC 协议
 让 AI 应用(Host，如 Claude Desktop)连接外部能力(Server 暴露的 Tools/Resources/Prompts)。
 本目录把 HF 课学到的 ML/NLP/LLM 能力(情感、NER、检索、生成)包成 MCP 服务器，供 Agent 调用。

 环境约束(本机)：官方 mcp / fastmcp SDK【已安装】。案例1-5 仍【手写 stdio + JSON-RPC 2.0】
 (参照 MCP Course/chapter2)是为了看懂协议底层；案例6 用【官方 fastmcp SDK】演示生产写法，
 两者说的是同一套 MCP 协议消息。datasets.load_dataset 用硬编码小语料兜底；用小模型
 (distilbert/bert-ner/MiniLM/SmolLM2)，mps 设备。每个案例都已 `python3 文件名` 亲测跑通。

 用法：python3 README_mcp实战导航.py         # 打印本导航
      python3 案例1_MCP工具服务器_NLP能力.py  # 跑任一案例(客户端会自动 spawn 服务器)
================================================================================
"""

CASES = [
    ("案例1_MCP工具服务器_NLP能力.py",
     "把情感分析(Ch1)+命名实体识别(Ch6/7)暴露成 tools(get_sentiment/extract_entities)。",
     "MCP 三步握手 initialize→tools/list(动态发现)→tools/call(真调用)；模型惰性加载；错误处理。"),
    ("案例2_MCP服务器_语义检索.py",
     "把 RAG 语义检索(Ch5/6)暴露成工具 search_knowledge(小知识库+嵌入+余弦 Top-K)。",
     "换说法也能命中(语义 vs 关键词)；Resources vs Tools 的选择；结果只回精简片段。"),
    ("案例3_MCP服务器_LLM生成.py",
     "把本地小 LLM 生成(Ch11, SmolLM2-135M-Instruct)暴露成工具 generate。",
     "chat template + generate；MCP 与 function calling 的关系；Sampling 原语；生成类工具的延迟坑。"),
    ("案例4_MCP_Agent多工具编排.py",
     "Agent 连接【多个】MCP 服务器(复用案例1+2)，自动发现→选工具→调用→综合回答。",
     "跨服务器聚合能力表 + 路由;规则式选工具(真实由 LLM function calling);多轮编排/命名空间/熔断。"),
    ("案例5_MCP生产部署.py",
     "一个服务器完整演示三大原语 Tools+Resources+Prompts;并讲生产部署全景。",
     "resources/read、prompts/get;stdio vs HTTP+SSE;鉴权(OAuth/RBAC);Claude Desktop 配置;FastMCP 对照。"),
    ("案例6_FastMCP官方SDK生产版.py",
     "同样把情感(Ch1)+NER(Ch6/7)暴露成工具，但用【官方 fastmcp SDK】——@mcp.tool 装饰器几行搞定。",
     "手写 vs FastMCP(同一套协议,SDK 自动生成 schema/处理传输);@mcp.resource;接入 Claude Desktop 配置。"),
    ("案例7_MCP协议进阶_Roots_Sampling_能力协商.py",
     "协议进阶：能力协商取交集、initialized 通知、Roots、Sampling、资源模板——用真实 JSON-RPC 消息演示。",
     "四类消息区分;server→client 反向请求(roots/list、sampling/createMessage);resources/templates/list。"),
    ("案例8_GradioMCP与官方SDK客户端.py",
     "Gradio launch(mcp_server=True) 一键把函数变 MCP 工具；官方 SDK/Agent 客户端(smolagents/hf Agent)写法。",
     "自动生成 schema + SSE 端点;生产里 Host 端不手写 subprocess，用 MCPClient/Agent 连接。"),
]

# MCP 原语 × HF 章节 的映射：哪一章的能力，用哪个 MCP 原语暴露最合适
MAPPING = [
    ("Ch1 情感分析",        "Tool",      "get_sentiment / analyze_text —— 有输入有计算，属'动作'", "案例1、案例5"),
    ("Ch2 微调分类器",       "Tool",      "把你微调的意图/情感分类器包成 tool 供 Agent 调用",       "(思路,见案例1)"),
    ("Ch5 语义检索/嵌入",    "Tool",      "search_knowledge —— 带 query 参数的检索动作",          "案例2、案例4"),
    ("Ch6 分词/NER",        "Tool",      "extract_entities —— token 分类+子词合并",             "案例1、案例4"),
    ("Ch7 token 分类微调",   "Tool",      "同 NER，微调后的实体识别器暴露为工具",                  "案例1"),
    ("Ch5/6 知识库文档",     "Resource",  "docs://faq —— 只读上下文，URI 寻址供模型阅读",          "案例5"),
    ("Ch11 LLM 生成",       "Tool",      "generate —— 本地小模型生成;或用 Sampling 借 Host 的 LLM", "案例3"),
    ("通用 提示模板",        "Prompt",    "summarize/triage —— 预置提示供用户一键选用",           "案例5"),
    ("Ch9/12 上线部署",     "Transport", "stdio(本地) / HTTP+SSE(远程) + 鉴权 + 多服务器",        "案例4、案例5"),
]

# MCP 核心概念速记(面试必背)
CONCEPTS = [
    ("MCP 是什么", "开放标准协议(JSON-RPC over stdio/HTTP)，让 AI 应用(Host)连外部能力(Server)。'AI 的 USB-C'。"),
    ("为什么需要", "解决 M 个应用×N 个工具的 M×N 定制集成爆炸 → 定标准后变 M+N：工具实现一次处处可连。"),
    ("四大原语",   "Tools(动作)/Resources(只读数据)/Prompts(提示模板)/Sampling(服务器反请 Host 的 LLM)。"),
    ("三方角色",   "Host(AI 应用) 内含 Client(每个连一个 Server)；Server 暴露原语。"),
    ("和 function calling 的关系",
                  "互补:function calling 是模型'决定调哪个工具、生成参数'(模型侧);MCP 是'工具怎么被描述/发现/连接/调用'(工具侧)。"),
    ("核心流程",   "initialize(握手/协商能力) → *_list(动态发现) → tools/call·resources/read·prompts/get(调用)。"),
]


def main():
    print(__doc__)
    print("─" * 72 + "\n 八个案例：\n" + "─" * 72)
    for i, (f, what, points) in enumerate(CASES, 1):
        print(f"\n[案例{i}] {f}")
        print(f"   做什么: {what}")
        print(f"   关键点: {points}")

    print("\n" + "─" * 72 + "\n MCP 原语 × HF 章节 映射：\n" + "─" * 72)
    print(f"   {'章节能力':<16}{'原语':<11}{'说明':<44}{'案例'}")
    for ch, prim, desc, where in MAPPING:
        print(f"   {ch:<16}{prim:<11}{desc:<44}{where}")

    print("\n" + "─" * 72 + "\n MCP 核心概念速记(面试)：\n" + "─" * 72)
    for k, v in CONCEPTS:
        print(f"   · {k}: {v}")

    print("\n" + "─" * 72)
    print(" 建议顺序：案例1(工具基础)→2(检索)→3(生成)→4(多服务器编排)→5(三原语+生产部署)"
          "→6(官方 FastMCP SDK)→7(协议进阶:Roots/Sampling/能力协商)→8(Gradio MCP+官方客户端)。")
    print(" 每个文件都可直接 `python3 文件名` 跑通;末尾都带【生产要点】+【面试题】。")


if __name__ == "__main__":
    main()
