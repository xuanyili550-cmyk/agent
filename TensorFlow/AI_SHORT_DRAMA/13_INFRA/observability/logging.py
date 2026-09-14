"""结构化日志。

生产上日志是被机器读的（Loki / CloudWatch / ELK），所以 ``log_json=True`` 时每行输出一个
JSON 对象，固定带 ``ts/level/logger/msg/request_id/task_id`` 字段；本地开发用普通文本格式。
``request_id_var`` 是 contextvar：API 中间件在请求入口塞进去，之后这条请求里所有日志自动带上，
不需要每个函数手动传。
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Any

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
task_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("task_id", default=None)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if request_id_var.get():
            payload["request_id"] = request_id_var.get()
        if task_id_var.get():
            payload["task_id"] = task_id_var.get()
        # logger.info("...", extra={"shot_id": ...}) 里的自定义字段原样带出
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get() or task_id_var.get()
        prefix = f"[{rid[:8]}] " if rid else ""
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name}: {prefix}{record.getMessage()}"
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}
        if extras:
            base += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


_configured = False


def configure_logging(level: str = "INFO", json_output: bool = False, force: bool = False) -> None:
    global _configured
    if _configured and not force:
        return
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    # 第三方库太吵，压到 WARNING
    for noisy in ("urllib3", "botocore", "httpx", "celery.utils.functional", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
