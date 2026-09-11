# 作品集实战项目（求职用）

学完 HuggingFace 课程后的 5 个可部署作品集项目。每个都：**真实模型 + Web 界面 + 可部署 + README + 对应一条简历 bullet**。
招聘看「做过什么」——每个项目都能给出公开链接/可复现。

| # | 项目 | 方向 | 技术栈 | 部署 |
|---|------|------|--------|------|
| 1 | [中文情感分析 API](项目1_中文情感分析API/) | 分类 + 服务化 | Transformers(中文) + Gradio + FastAPI | HF Spaces |
| 2 | [文档问答 RAG](项目2_文档问答RAG/) | RAG(最热) | bge-small-zh + ChromaDB + mlx-lm | HF Spaces* |
| 3 | [本地 LLM 推理服务](项目3_本地LLM推理服务/) | LLM 服务 | mlx-lm + OpenAI 兼容 FastAPI | 本地(Mac) |
| 4 | [多功能 NLP 工具箱](项目4_多功能NLP工具箱/) | 多任务集成 | 情感/NER/翻译/摘要 + Gradio Tabs | HF Spaces |
| 5 | [LLM Agent](项目5_LLM_Agent/) | Agent/function calling(前沿) | mlx-lm + 工具调用 | 本地(Mac) |
| 6 | [智能客服助手](项目6_智能客服助手/) | 集大成(情感+NER+RAG) | 中文情感/NER + bge + Gradio | HF Spaces |

\* 含本地 LLM(mlx-lm) 的项目在 Mac(Apple Silicon)上真跑；上云/Spaces(Linux) 时把 mlx-lm 换成
vLLM/HF InferenceClient(API 契约一致，前端不改)——见 `实战练习/chapter实战/生产05`。

## 通用约定
- 每个项目 `python3 app.py smoke` 只做自检(不起服务)；`python3 app.py` 起 Web；部分有 `api` 模式。
- 中文任务用中文模型(英文模型喂中文会失准)。
- 生产结构：模型单例加载、输入校验(gr.Error)、队列并发、流式(适用时)、FastAPI 挂载、部署说明。

## 每个项目对应的简历 bullet
1. 部署中文情感分析在线服务(Transformers + Gradio + FastAPI REST API)。
2. 构建文档问答 RAG 系统(bge 中文嵌入 + ChromaDB 向量库 + 本地 LLM 生成，可溯源)。
3. 实现 OpenAI 兼容的本地 LLM 推理服务(mlx-lm，流式，概念对齐 vLLM)。
4. 开发多任务 NLP 工具箱(情感/实体/翻译/摘要，多标签页产品化)。
5. 实现能调用工具的 LLM Agent(function calling，本地 LLM 大脑 + NLP 工具编排)。
6. 开发并部署智能客服助手(情感分流 + 实体路由 + RAG-FAQ + 自动回复/转人工决策)。
