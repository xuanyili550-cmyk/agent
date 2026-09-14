# 多阶段：base 装系统依赖 + 全部 Python 依赖；api 与 worker 共用同一镜像，靠 command 区分。
# 之前的镜像只 COPY 了 08/09/13 三个目录，worker 一起来就 ImportError（llm_task 要 02，
# image_task 要 07）——现在把所有顶层模块都拷进去。
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# ffmpeg 带 libass（字幕硬烧）；libgl/glib 给 opencv；fonts-noto-cjk 让中文字幕能渲染
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先只拷 requirements，利用 Docker 层缓存：代码改动不会重装依赖
COPY requirements.txt ./requirements.txt
COPY 13_INFRA/requirements-infra.txt ./13_INFRA/requirements-infra.txt
COPY 08_QC/requirements-qc.txt ./08_QC/requirements-qc.txt
COPY 09_POST/requirements-post.txt ./09_POST/requirements-post.txt

# INSTALL_ML=1 时装 torch/transformers/diffusers（GPU worker 镜像），API 镜像用 0 保持精简
ARG INSTALL_ML=0
RUN pip install -r 13_INFRA/requirements-infra.txt -r 09_POST/requirements-post.txt \
    && if [ "$INSTALL_ML" = "1" ]; then pip install -r requirements.txt -r 08_QC/requirements-qc.txt; \
       else pip install pydantic langgraph jinja2 anthropic openai pillow pandas scipy pyyaml requests \
            google-api-python-client google-auth-oauthlib; fi

COPY 01_CONTENT ./01_CONTENT
COPY 02_STORY_ENGINE ./02_STORY_ENGINE
COPY 03_STRUCTURED_DATA ./03_STRUCTURED_DATA
COPY 04_DATASET ./04_DATASET
COPY 06_MODELS ./06_MODELS
COPY 07_GENERATION ./07_GENERATION
COPY 08_QC ./08_QC
COPY 09_POST ./09_POST
COPY 10_EPISODES ./10_EPISODES
COPY 11_PUBLISH ./11_PUBLISH
COPY 12_ANALYTICS ./12_ANALYTICS
COPY 13_INFRA ./13_INFRA

RUN useradd --create-home --uid 10001 app && mkdir -p /app/artifacts /app/storage_data && chown -R app:app /app
USER app

EXPOSE 8000 9100

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)" || exit 1

CMD ["uvicorn", "13_INFRA.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
