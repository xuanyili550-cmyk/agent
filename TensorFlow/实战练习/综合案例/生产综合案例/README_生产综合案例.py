"""
================================================================================
 生产综合案例导航 · 5 个真实生产案例，覆盖 全部课程功能
================================================================================
 5 个案例【合起来】覆盖 MCP Course + Agent Course + Chapter 1/2/3/5/6/7/9/11/12 的全部功能，
 每个都是【真实生产案例】：真实模型、模型单例、Gradio Web、输入校验(gr.Error)、队列、
 (适用时)FastAPI 挂载/OpenAI 兼容 API、可部署。中文任务用中文模型。

 —— 5 个案例 ——
   生产案例1_内容智能中台.py          情感/实体/翻译/摘要/完形填空/分词 六能力 + Web + API
   生产案例2_RAG_Agent知识平台.py     ChromaDB-RAG + MCP 工具 + Agent 路由 + LLM 生成 + Web
   生产案例3_微调与对齐平台.py         分类微调 + LoRA + GRPO 三条【生产 GPU 训练】(trl Trainer/SFT/GRPO,需GPU) + Web
   生产案例4_LLM推理服务与Agent网关.py  mlx-lm 流式 + OpenAI 兼容 API + MCP 工具 + Agent + 聊天 Web
   生产案例5_智能客服全栈.py           情感+NER脱敏+RAG+Agent决策+LLM回复(客服集大成) + Web

 —— 功能覆盖矩阵(每列是章节，√ 表示该案例覆盖) ——
   案例   Ch1 Ch2 Ch3 Ch5 Ch6 Ch7 Ch9 Ch11 Ch12 MCP Agent
   案例1   √       √       √   √   √
   案例2               √           √              √   √
   案例3       √                   √        √    √
   案例4   √(推理)                         √               √    √
   案例5   √           √       √   √                        √
   —— 每列都至少被一个案例覆盖：全课程功能齐 ——

 —— 说明 ——
   · 含本地 LLM(mlx-lm)的案例(2/4/5)在 Mac(Apple Silicon)真跑；上云换 vLLM/InferenceClient(见 ../../chapter实战/生产05)。
   · Agent 编排用确定性路由(本机小模型选工具不可靠)；生产换大模型 + smolagents/function calling。
   · 每个 `python3 案例.py smoke` 自检；`python3 案例.py` 起 Web；部分有 `api` 模式。
================================================================================
"""
COVER = {
    "案例1_内容智能中台": ["Ch1", "Ch3", "Ch6", "Ch7", "Ch9"],
    "案例2_RAG_Agent知识平台": ["Ch5", "Ch9", "MCP", "Agent"],
    "案例3_微调与对齐平台": ["Ch2", "Ch9", "Ch11", "Ch12"],
    "案例4_LLM推理服务与Agent网关": ["Ch1", "Ch9", "MCP", "Agent"],
    "案例5_智能客服全栈": ["Ch1", "Ch5", "Ch7", "Ch9", "Agent"],
}

if __name__ == "__main__":
    print(__doc__)
    allc = ["Ch1", "Ch2", "Ch3", "Ch5", "Ch6", "Ch7", "Ch9", "Ch11", "Ch12", "MCP", "Agent"]
    covered = set(c for v in COVER.values() for c in v)
    print(" 覆盖校验：", {c: ("√" if c in covered else "✗") for c in allc})
    assert covered.issuperset(allc), "有章节未覆盖！"
    print(" ✅ 全部章节/课程功能均被 5 个案例覆盖。")
