# 本地 LLM 推理服务（OpenAI 兼容 API + 聊天 Web）

在 Mac(Apple Silicon)上用 **mlx-lm** 跑一个本地 LLM，暴露成 **OpenAI 兼容的 `/v1/chat/completions` API**
+ 一个流式聊天 Web。概念对齐生产的 vLLM，本机用 mlx-lm 替代（Mac 上装不了 vLLM）。

## 功能特性（生产级）
- **OpenAI 兼容**：`POST /v1/chat/completions`（非流式 + 流式 SSE），客户端/前端无缝切换、不锁厂商。
- **流式输出**：打字机效果（`stream_generate` 逐块 yield / SSE）。
- **模型单例**：只加载一次常驻。
- **聊天 Web**：Gradio ChatInterface（Gradio 6：history 默认就是 messages 字典列表）。
- **健康探针**：`GET /health`。

## 运行
```bash
pip install -r requirements.txt
python3 app.py smoke   # 加载+生成(非流式/流式)自检
python3 app.py         # 起聊天 Web
python3 app.py api     # 起 OpenAI 兼容服务(端口 8000)
```
用 OpenAI SDK 调用本地服务：
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
print(c.chat.completions.create(model="local", messages=[{"role":"user","content":"你好"}]).choices[0].message.content)
```

## 部署
- **本地/Mac**：直接跑（mlx-lm 用 Apple Silicon 的统一内存，很省）。
- **上云**：mlx 仅 Apple 芯片 → 云端把 mlx-lm 换成 **vLLM**（`python -m vllm.entrypoints.openai.api_server ...`），
  **API 契约完全一样，前端/客户端不用改**（这正是 OpenAI 兼容的价值）。见 `实战练习/chapter实战/生产01、生产05`。

## 技术栈
mlx-lm(Qwen2.5-0.5B-Instruct-4bit) · FastAPI · Gradio 6

## 简历 bullet
> 实现 OpenAI 兼容的本地 LLM 推理服务（mlx-lm + FastAPI），支持流式输出与标准 `/v1/chat/completions`，
> 前端可无缝切换到 vLLM/云端，概念与生产推理栈对齐。
