"""
================================================================================
 Agent 实战导航 · 把 HF Agents 课程 + Chapter1-12(NLP能力) + MCP 串成生产级 Agent
================================================================================
 主题：Agent = LLM(大脑) + Tools(手脚) + 循环(ReAct:推理→行动→观察)。本目录把
 “Agent 课学到的框架(smolagents 为主)”和“前面章节做出来的真实能力”整合成能落地的智能体：
   · Chapter1-12 的 NLP 能力(中文情感/NER/语义检索/生成) → 包成 Agent 的【工具】
   · MCP 服务器(mcp实战) → Agent 通过 MCP 协议【发现并调用】远程工具
   · Agentic RAG、多工具编排、多框架对比、生产部署与可观测

 —— 本机现实(务必知道) ——
   · smolagents 的“配线”本机能跑(TransformersModel 本地模型)，但【小模型(135M/1.7B)太弱，
     当不了代码智能体】——生成的代码常语法错误、选错工具(已实测)。所以：
       工具 → 全部本地【真跑 + 自检】(用缓存的中文 NLP 模型)；
       Agent 大脑 → 生产用 InferenceClientModel(HF Token) / OpenAIServerModel(vLLM，见 chapter实战/生产05)；
       本地 end-to-end → 用【确定性规则路由】跑同一批工具，证明编排流程(真实生产由 LLM function calling)。
   · 已装框架：smolagents / langgraph / langchain / llama_index / duckduckgo_search。

 —— 文件 ——
   案例1_smolagents基础_工具与Agent.py   @tool / Tool 子类 / CodeAgent vs ToolCallingAgent / 模型后端
   案例2_NLP能力Agent_中文客服.py         把中文情感/NER/检索(Ch1/5/6/7)包成工具，Agent 处理工单
   案例3_Agentic_RAG_中文知识库.py        检索工具(bge-small-zh 向量) + Agent 先检索再回答(可溯源)
   案例4_Agent接MCP工具.py               smolagents 通过 MCP 连 mcp实战 的 NLP 服务器，自动发现工具
   案例5_多框架对比_smolagents_LangGraph_LlamaIndex.py  同一任务三框架骨架对比
   生产_Agent部署_模型后端与观测.py       InferenceClientModel/vLLM + GradioUI + Langfuse 可观测 + 分享

 —— 核心概念速记(面试) ——
   · ReAct vs CoT：CoT 只在脑内逐步推理；ReAct = 推理 + 调外部工具(行动)+ 看结果(观察)，能查实时信息。
   · CodeAgent vs ToolCallingAgent：前者让 LLM【写 Python 代码】调工具(更灵活)；后者让 LLM 输出【JSON 工具调用】(更稳)。
   · 模型后端：TransformersModel(本地) / InferenceClientModel(HF/供应商) / OpenAIServerModel(vLLM/兼容) / LiteLLMModel。
   · Tool 三要素：name + description + inputSchema(类型) —— LLM 靠这些“读懂”工具、决定调不调、传什么参。
   · 记忆：smolagents 靠 reset=False；llama-index 传 Context；langgraph 接 messages/MemorySaver。
   · 可观测：OpenTelemetry + Langfuse(SmolagentsInstrumentor)，看每步 推理/工具调用/token/延迟。
================================================================================
"""

MAPPING = [
    ("Ch1 情感",        "get_sentiment 工具",   "Agent 判断工单情绪 → 决定优先级/是否转人工", "案例2"),
    ("Ch6/7 NER",       "extract_entities 工具", "Agent 抽实体做路由/脱敏",                 "案例2"),
    ("Ch5 语义检索",     "search_kb 工具",       "Agentic RAG：Agent 决定检索什么、检索几次",  "案例3"),
    ("Ch11 生成",        "LLM 大脑本身",         "Agent 的推理/回复由 LLM(可微调版)驱动",      "全部"),
    ("MCP(mcp实战)",     "ToolCollection.from_mcp", "Agent 连 MCP 服务器动态发现工具(即插即用)", "案例4"),
    ("生产05 vLLM",      "OpenAIServerModel",    "Agent 大脑接自建 vLLM(不锁厂商、可私有化)",   "生产"),
]

if __name__ == "__main__":
    print(__doc__)
    print("─" * 74 + "\n Agent 工具 × 前面章节能力 映射：\n" + "─" * 74)
    print(f"   {'章节能力':<16}{'包成的工具/角色':<24}{'Agent 怎么用':<34}{'案例'}")
    for cap, tool, how, where in MAPPING:
        print(f"   {cap:<16}{tool:<24}{how:<34}{where}")
    print("\n 建议顺序：案例1(基础)→2(NLP工具)→3(Agentic RAG)→4(接MCP)→5(多框架)→生产(部署+观测)。")
    print(" 每个文件 `python3 文件名 smoke` 只跑工具自检(快)；`python3 文件名` 看更多(部分需 HF Token/服务)。")
