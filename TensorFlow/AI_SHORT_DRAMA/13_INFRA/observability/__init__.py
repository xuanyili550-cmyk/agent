"""可观测性包：结构化日志（logging.py）+ Prometheus 指标（metrics.py）的统一导出。

队列任务、API 中间件、编排层都从这里 import，不直接依赖 prometheus_client。
"""

from .logging import configure_logging, get_logger, request_id_var
from .metrics import (
    AGENT_RUNS,
    AGENT_TOOL_ROUNDS,
    HTTP_REQUESTS,
    LLM_COST_USD,
    LLM_FALLBACK,
    LLM_TOKENS,
    PIPELINE_RUNS,
    QC_DECISIONS,
    RETRY_LADDER,
    TASK_DURATION,
    TASK_IN_PROGRESS_SAFE,
    TASK_TOTAL,
    TOOL_CALLS,
    TOOL_DURATION,
    four_key_metrics,
    metrics_response,
    observe_task,
    record_agent_run,
    record_tool_call,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "request_id_var",
    "AGENT_RUNS",
    "AGENT_TOOL_ROUNDS",
    "HTTP_REQUESTS",
    "LLM_COST_USD",
    "LLM_FALLBACK",
    "LLM_TOKENS",
    "PIPELINE_RUNS",
    "QC_DECISIONS",
    "RETRY_LADDER",
    "TASK_DURATION",
    "TASK_IN_PROGRESS_SAFE",
    "TASK_TOTAL",
    "TOOL_CALLS",
    "TOOL_DURATION",
    "four_key_metrics",
    "metrics_response",
    "observe_task",
    "record_agent_run",
    "record_tool_call",
]
