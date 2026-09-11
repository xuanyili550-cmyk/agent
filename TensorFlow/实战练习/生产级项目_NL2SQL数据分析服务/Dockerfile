# 多阶段构建：builder 装依赖 → runtime 精简镜像
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
# 生产镜像用 openai 后端(连 vLLM/Ollama)，不装 mlx(仅 Mac)
RUN grep -v "mlx-lm" requirements.txt > req.txt \
    && pip install --no-cache-dir --prefix=/install -r req.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /install /usr/local
COPY app ./app
ENV APP_LLM_BACKEND=openai
EXPOSE 8000
# 非 root 运行(安全)
RUN useradd -m appuser && chown -R appuser /app
USER appuser
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import httpx;httpx.get('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
