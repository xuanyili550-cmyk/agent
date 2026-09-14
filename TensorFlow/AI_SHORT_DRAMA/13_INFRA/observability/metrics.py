"""Prometheus 指标。

API 进程通过 ``GET /metrics`` 暴露；Celery worker 在 ``worker_process_init`` 时起一个
``prometheus_client.start_http_server``（端口 ``WORKER_METRICS_PORT``），13_INFRA/monitoring/
prometheus.yml 直接抓这些端口。指标命名遵循 Prometheus 约定：``_total`` 计数、``_seconds`` 直方图。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

TASK_TOTAL = Counter("drama_task_total", "队列任务次数", ["task", "status"])
TASK_DURATION = Histogram(
    "drama_task_duration_seconds",
    "队列任务耗时",
    ["task"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800),
)
TASK_IN_PROGRESS = Gauge("drama_task_in_progress", "正在执行的任务数", ["task"])
LLM_TOKENS = Counter("drama_llm_tokens_total", "LLM token 用量", ["provider", "model", "kind"])
LLM_COST_USD = Counter("drama_llm_cost_usd_total", "LLM 估算花费（美元）", ["provider", "model"])
QC_DECISIONS = Counter("drama_qc_decisions_total", "QC 判定次数", ["decision"])
RETRY_LADDER = Counter("drama_retry_ladder_total", "三级重试阶梯触发次数", ["level"])
HTTP_REQUESTS = Counter("drama_http_requests_total", "HTTP 请求数", ["method", "path", "status"])
HTTP_LATENCY = Histogram("drama_http_request_duration_seconds", "HTTP 请求耗时", ["method", "path"])
PIPELINE_RUNS = Counter("drama_pipeline_runs_total", "流水线运行状态变更", ["status"])


def TASK_IN_PROGRESS_SAFE(task_name: str, delta: int) -> None:
    """Gauge 的 inc/dec 封装，任务名为 None（eager 模式下偶发）时不报错。"""
    if not task_name:
        return
    if delta > 0:
        TASK_IN_PROGRESS.labels(task_name).inc(delta)
    else:
        TASK_IN_PROGRESS.labels(task_name).dec(-delta)


@contextmanager
def observe_task(task_name: str) -> Iterator[None]:
    TASK_IN_PROGRESS.labels(task_name).inc()
    started = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "failure"
        raise
    finally:
        TASK_DURATION.labels(task_name).observe(time.perf_counter() - started)
        TASK_TOTAL.labels(task_name, status).inc()
        TASK_IN_PROGRESS.labels(task_name).dec()


def metrics_response() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
