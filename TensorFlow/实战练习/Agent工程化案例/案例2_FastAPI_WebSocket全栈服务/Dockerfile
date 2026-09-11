# 全栈 Agent 服务 · 后端镜像
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# 起 FastAPI(含 WebSocket)。生产在 Linux 上用 vLLM/Ollama 作 LLM 后端(mlx 仅 Apple 芯片)。
CMD ["uvicorn", "app:make_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
