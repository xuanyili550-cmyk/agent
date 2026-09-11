FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY 08_QC/requirements-qc.txt ./08_QC/requirements-qc.txt
COPY 09_POST/requirements-post.txt ./09_POST/requirements-post.txt
COPY 13_INFRA/requirements-infra.txt ./13_INFRA/requirements-infra.txt

RUN pip install --no-cache-dir \
    -r 08_QC/requirements-qc.txt \
    -r 13_INFRA/requirements-infra.txt

COPY 08_QC ./08_QC
COPY 09_POST ./09_POST
COPY 13_INFRA ./13_INFRA

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "13_INFRA.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
