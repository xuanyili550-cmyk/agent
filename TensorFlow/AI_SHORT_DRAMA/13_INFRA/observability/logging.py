"""结构化日志。

生产上日志是被机器读的（Loki / CloudWatch / ELK），所以 ``log_json=True`` 时每行输出一个
JSON 对象，固定带 ``ts/level/logger/msg/request_id/task_id`` 字段；本地开发用普通文本格式。
``request_id_var`` 是 contextvar：API 中间件在请求入口塞进去，之后这条请求里所有日志自动带上，
不需要每个函数手动传。

API 进程在 main.py 启动时、worker 进程在 celery_app 的 ``setup_logging`` 信号里各调一次 ``configure_logging``，
两边输出格式完全一致，便于在日志系统里按 request_id / task_id 把一次请求触发的整条链路串起来。
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Any

# 请求/任务级上下文：由中间件（API）和 BaseTask（worker）设置，Formatter 读取
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
task_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("task_id", default=None)

# LogRecord 自带的属性名集合：用来区分 logging 内部字段和调用方通过 extra= 传进来的自定义字段
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """每条日志输出一行 JSON（生产用）。"""

    def format(self, record: logging.LogRecord) -> str:
        """组装固定字段 + 上下文 id + extra 自定义字段 + 异常栈，序列化成单行 JSON。

        ``ensure_ascii=False`` 让中文日志可读；``default=str`` 兜底不可序列化的对象（如 Path、datetime），避免打日志本身抛异常。
        """
        payload: dict[str, Any] = {
            # UTC ISO 时间 + 毫秒 + Z 后缀，日志系统按此解析时间戳
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
    """人眼可读的单行文本格式（本地开发用），extra 字段以 ``k=v`` 追加在消息后。"""

    def format(self, record: logging.LogRecord) -> str:
        """``HH:MM:SS LEVEL logger: [rid前8位] 消息 k=v ...``，有异常时换行追加栈。"""
        rid = request_id_var.get() or task_id_var.get()
        # 只取前 8 位：终端里够区分请求，又不至于把每行撑得太长
        prefix = f"[{rid[:8]}] " if rid else ""
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name}: {prefix}{record.getMessage()}"
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}
        if extras:
            base += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


# 防止重复配置：多次 import / 多次调用只装一次 handler，否则每条日志会打印多遍
_configured = False


def configure_logging(level: str = "INFO", json_output: bool = False, force: bool = False) -> None:
    """给 root logger 装唯一的 stdout handler，并按 ``json_output`` 选格式。

    ``force=True`` 用于 Celery 的 ``setup_logging`` 信号：Celery 会先自己配一套 handler，我们要覆盖它，
    所以即使之前已经配置过也要重来一遍。输出到 stdout 而不是文件，是容器化部署的惯例（由采集器收集）。
    """
    global _configured
    if _configured and not force:
        return
    root = logging.getLogger()
    # 清掉已有 handler（包括 Celery/uvicorn 装的），保证只有一份输出
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
    """取命名 logger 的薄封装；各模块统一从这里拿，方便以后集中加过滤器或改实现。"""
    return logging.getLogger(name)
