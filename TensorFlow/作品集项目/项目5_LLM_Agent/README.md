# LLM Agent / Function Calling（能调工具的小智能体）

一个能【自己决定调用哪个工具】的小 Agent：本地 LLM(mlx-lm)当大脑，把中文 NLP 能力 + 计算器当工具，
走 **选工具 → 调用 → 综合回答**（function calling / ReAct 的核心）。前沿方向。

## 功能特性
- **工具集**（本地真跑）：`tool_sentiment`(中文情感) / `tool_search_faq`(bge 中文检索) / `tool_calculator`(安全算术)。
- **LLM 大脑**：mlx-lm(Qwen2.5-0.5B-Instruct-4bit)提议工具+参数。
- **规则校验兜底**：小模型可靠性有限——LLM 提议不合理(参数为空/算术无数字等)时自动按关键词兜底，
  保证 Demo 稳定（生产换大模型/smolagents 可去掉兜底）。
- **过程可见**：Web 上显示「🧠 选工具 → 🔧 调用+观察 → 💬 回答」全过程。

## 运行
```bash
pip install -r requirements.txt
python3 app.py smoke   # 三类问题(算术/FAQ/情感)各跑一遍
python3 app.py         # 起聊天 Web，看 Agent 的工具调用过程
```

## 生产升级
换 **InferenceClientModel / vLLM 大模型 + smolagents 框架**（见 `实战练习/Agent实战`），
工具契约不变，Agent 推理更强、可去掉关键词兜底、支持多步 ReAct 与多工具编排。

## 部署
- **本地/Mac**：直接跑（mlx-lm）。
- **上云**：把 `_llm` 换成 vLLM/InferenceClient；工具与编排逻辑不变。

## 技术栈
mlx-lm · Transformers(中文情感/bge) · Gradio 6

## 简历 bullet
> 实现能调用工具的 LLM Agent（function calling）：本地 LLM 大脑 + 中文 NLP/计算器工具，
> 完成「选工具→调用→综合回答」编排，含规则校验兜底保证稳定性；可平滑升级到 smolagents + vLLM。
