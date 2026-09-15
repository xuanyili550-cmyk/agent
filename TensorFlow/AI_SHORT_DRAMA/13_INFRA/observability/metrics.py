"""Prometheus 指标。

API 进程通过 ``GET /metrics`` 暴露；Celery worker 在 ``worker_process_init`` 时起一个
``prometheus_client.start_http_server``（端口 ``WORKER_METRICS_PORT``），13_INFRA/monitoring/
prometheus.yml 直接抓这些端口。指标命名遵循 Prometheus 约定：``_total`` 计数、``_seconds`` 直方图。

《Agent 搭建指南》要求的"四大监控指标"在这里的落点：
    1. Token 消耗       -> LLM_TOKENS / LLM_COST_USD（按 provider/model 分）
    2. 工具调用成功率   -> TOOL_CALLS（按 tool/status 分），success / 总数
    3. 执行时间         -> TASK_DURATION（队列任务）、TOOL_DURATION（单个工具）、AGENT_TOOL_ROUNDS（一次 run 用了几轮）
    4. 错误率           -> TASK_TOTAL 里 failure / (success + failure + retry)，LLM_FALLBACK 记录降级次数
``four_key_metrics()`` 把这四项从 Prometheus registry 里汇总成一个 dict，``GET /metrics/summary``
直接返回，不用先搭 Grafana 也能看到当前状态。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Gauge, Histogram, generate_latest

# ---- 队列任务：执行时间 / 错误率 ----
TASK_TOTAL = Counter("drama_task_total", "队列任务次数", ["task", "status"])
TASK_DURATION = Histogram(
    "drama_task_duration_seconds",
    "队列任务耗时",
    ["task"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800),
)
TASK_IN_PROGRESS = Gauge("drama_task_in_progress", "正在执行的任务数", ["task"])

# ---- LLM：token 消耗 / 成本 / 降级 ----
LLM_TOKENS = Counter("drama_llm_tokens_total", "LLM token 用量", ["provider", "model", "kind"])
LLM_COST_USD = Counter("drama_llm_cost_usd_total", "LLM 估算花费（美元）", ["provider", "model"])
LLM_FALLBACK = Counter("drama_llm_fallback_total", "多模型降级次数（主模型不可用切到备用）", ["from_model", "to_model"])

# ---- Agent 工具调用：成功率 / 耗时 / 轮数 ----
TOOL_CALLS = Counter("drama_tool_calls_total", "Agent 工具调用次数", ["tool", "status"])
TOOL_DURATION = Histogram("drama_tool_call_duration_seconds", "单次工具调用耗时", ["tool"], buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60))
AGENT_RUNS = Counter("drama_agent_runs_total", "ToolAgent.run() 次数（按停止原因）", ["agent", "stopped_reason"])
AGENT_TOOL_ROUNDS = Histogram("drama_agent_tool_rounds", "一次 Agent run 调用了几轮工具", ["agent"], buckets=(0, 1, 2, 3, 4, 5, 8, 10))

# ---- 业务 ----
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
    """把一段代码当成一个"任务"计时 + 计成败（BaseTask 之外的地方，例如脚本里手动跑阶段时用）。"""
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


def record_tool_call(tool: str, status: str, duration_seconds: float) -> None:
    """一次工具调用 -> TOOL_CALLS + TOOL_DURATION（ToolAgent.tool_call_sink 接这里）。"""
    TOOL_CALLS.labels(tool, status).inc()
    TOOL_DURATION.labels(tool).observe(duration_seconds)


def record_agent_run(agent: str, stopped_reason: str, tool_rounds: int) -> None:
    """一次 ToolAgent.run() -> AGENT_RUNS + AGENT_TOOL_ROUNDS。"""
    AGENT_RUNS.labels(agent, stopped_reason).inc()
    AGENT_TOOL_ROUNDS.labels(agent).observe(tool_rounds)


def metrics_response() -> tuple[bytes, str]:
    """Prometheus 文本格式的全部指标，给 GET /metrics 用。"""
    return generate_latest(), CONTENT_TYPE_LATEST


# --------------------------------------------------------------------------------------
# 四大监控指标汇总
# --------------------------------------------------------------------------------------


def _samples(metric_name: str) -> list[tuple[dict[str, str], float]]:
    """从 registry 里取某个指标的全部样本 (labels, value)。

    prometheus_client 暴露给抓取器的名字带 ``_total`` / ``_sum`` / ``_count`` 后缀，而 ``collect()``
    里 Counter 的 family 名不带 ``_total``；这里按样本名精确匹配，调用方传完整样本名。
    """
    out: list[tuple[dict[str, str], float]] = []
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name == metric_name:
                out.append((dict(sample.labels), float(sample.value)))
    return out


def _rate(numerator: float, denominator: float) -> float | None:
    """安全除法：分母为 0 返回 None（"还没有数据"和"0%"是两回事）。"""
    return round(numerator / denominator, 4) if denominator else None


def four_key_metrics() -> dict[str, Any]:
    """把四大监控指标从 Prometheus registry 汇总成一个 JSON 友好的 dict。

    只做汇总不做时间窗口：Prometheus/Grafana 负责"最近 5 分钟"这类查询，这里给的是进程启动以来
    的累计值，用于快速自检和没有 Prometheus 的本地开发环境。
    """
    # 1. token 消耗
    tokens: dict[str, dict[str, float]] = {}
    for labels, value in _samples("drama_llm_tokens_total"):
        key = f"{labels['provider']}/{labels['model']}"
        tokens.setdefault(key, {"input": 0.0, "output": 0.0, "cost_usd": 0.0})
        tokens[key][labels["kind"]] += value
    for labels, value in _samples("drama_llm_cost_usd_total"):
        key = f"{labels['provider']}/{labels['model']}"
        tokens.setdefault(key, {"input": 0.0, "output": 0.0, "cost_usd": 0.0})["cost_usd"] += value
    total_tokens = sum(v["input"] + v["output"] for v in tokens.values())

    # 2. 工具调用成功率
    tool_calls: dict[str, dict[str, float]] = {}
    for labels, value in _samples("drama_tool_calls_total"):
        bucket = tool_calls.setdefault(labels["tool"], {})
        bucket[labels["status"]] = bucket.get(labels["status"], 0.0) + value
    tool_total = sum(sum(s.values()) for s in tool_calls.values())
    tool_success = sum(s.get("success", 0.0) for s in tool_calls.values())

    # 3. 执行时间：直方图的 sum/count 给平均值
    durations: dict[str, dict[str, float | None]] = {}
    sums = {labels["task"]: value for labels, value in _samples("drama_task_duration_seconds_sum")}
    counts = {labels["task"]: value for labels, value in _samples("drama_task_duration_seconds_count")}
    for task, count in counts.items():
        durations[task] = {"count": count, "total_seconds": round(sums.get(task, 0.0), 3), "avg_seconds": _rate(sums.get(task, 0.0), count)}

    # 4. 错误率：failure / (success + failure + retry)，按任务分
    task_status: dict[str, dict[str, float]] = {}
    for labels, value in _samples("drama_task_total"):
        task_status.setdefault(labels["task"], {})[labels["status"]] = value
    task_all = sum(sum(s.values()) for s in task_status.values())
    task_failure = sum(s.get("failure", 0.0) for s in task_status.values())
    fallbacks = sum(value for _, value in _samples("drama_llm_fallback_total"))
    agent_runs: dict[str, float] = {}
    for labels, value in _samples("drama_agent_runs_total"):
        agent_runs[labels["stopped_reason"]] = agent_runs.get(labels["stopped_reason"], 0.0) + value

    return {
        "token_usage": {"total_tokens": total_tokens, "by_model": tokens},
        "tool_calls": {"total": tool_total, "success": tool_success, "success_rate": _rate(tool_success, tool_total), "by_tool": tool_calls},
        "execution_time": {"by_task": durations},
        "error_rate": {
            "tasks_total": task_all,
            "tasks_failed": task_failure,
            "error_rate": _rate(task_failure, task_all),
            "by_task": task_status,
            "llm_fallbacks": fallbacks,
            "agent_runs_by_stop_reason": agent_runs,
        },
    }
