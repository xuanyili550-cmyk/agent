"""所有队列任务的基类：自动重试 + 指标 + 结构化日志上下文。

- ``autoretry_for``：只对"瞬时"错误自动重试（网络、限流、5xx、LLM/provider 标记的 Transient*），
  指数退避 + 抖动，最多 ``max_retries`` 次。这就是三级重试阶梯的第一级"同样参数再试一次"
  在任务层的实现；业务错误（prompt 被拒、校验失败）不在此列，由任务体内的阶梯逻辑处理。
- ``acks_late=True`` + ``reject_on_worker_lost=True``：worker 被 OOM 杀掉时消息回到队列，不丢任务。
- 每个任务的耗时/成败进 Prometheus；task_id 进日志 contextvar，任务内所有日志自动带上。

在平台里的位置：workers/celery_app.py 把它设为 ``celery_app.Task``，所以所有 ``@celery_app.task`` 定义的任务
都自动继承这套行为，任务模块本身不需要写任何重试/指标代码。
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from celery import Task

# eager 模式退避的时间缩放：测试里设成 0 立刻重试，本地开发默认 0.1 倍
EAGER_BACKOFF_SCALE = float(os.environ.get("EAGER_BACKOFF_SCALE", "0.1"))

# 02/07 是数字开头的顶层包，只能 importlib 按名字加载；先保证项目根在 sys.path 上
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ..config import get_settings  # noqa: E402
from ..observability import TASK_DURATION, TASK_IN_PROGRESS_SAFE, TASK_TOTAL, get_logger  # noqa: E402
from ..observability.logging import task_id_var  # noqa: E402

log = get_logger("queue")


def _transient_exceptions() -> tuple[type[BaseException], ...]:
    """把 02/07 里定义的瞬时错误类收集起来（延迟 import，避免 worker 启动时就加载 torch）。"""
    exceptions: list[type[BaseException]] = [ConnectionError, TimeoutError, requests.ConnectionError, requests.Timeout]
    for module_name, attr in (("02_STORY_ENGINE.agents.base", "TransientLLMError"), ("07_GENERATION.errors", "TransientProviderError")):
        try:
            exceptions.append(getattr(importlib.import_module(module_name), attr))
        except Exception:  # 模块不可用（例如镜像里没装该层）时跳过
            pass
    try:
        import redis

        exceptions.append(redis.exceptions.ConnectionError)
    except Exception:
        pass
    return tuple(exceptions)


class BaseTask(Task):
    """Celery 任务基类：把重试策略、Prometheus 指标和日志上下文集中在 ``__call__`` 里，子类只写 ``run``。

    类属性说明：
    - abstract：本身不注册成任务，只做基类；
    - acks_late / reject_on_worker_lost：任务执行完才 ack，worker 崩溃时消息回队列重投；
    - retry_backoff / retry_backoff_max / retry_jitter：指数退避（最长 600s）加随机抖动，避免一群任务同时重试打爆上游；
    - autoretry_for 初始为空元组，首次调用时才解析成真正的瞬时错误类（见 ``__call__``），因为解析要 import 02/07。
    """

    abstract = True
    acks_late = True
    reject_on_worker_lost = True
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    autoretry_for: tuple = ()
    _autoretry_resolved = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """任务入口：解析重试配置 -> 绑定 task_id 到日志 -> 执行 -> 按结果分类计数。

        瞬时错误交给 ``self.retry``（Celery 按退避策略重新投递）；其它异常记 failure 后原样抛出，
        让 Celery 把任务标成 FAILURE。eager 模式（测试/本地）没有 broker，走进程内重试。
        """
        # 首次调用时才解析瞬时错误类，并把 settings 里的重试次数带进来
        if not type(self)._autoretry_resolved:
            type(self).autoretry_for = _transient_exceptions()
            type(self).max_retries = get_settings().task_max_retries
            type(self)._autoretry_resolved = True
        # contextvar 里放 task_id，任务体内所有 get_logger 打的日志都会自动带上它
        token = task_id_var.set(self.request.id)
        TASK_IN_PROGRESS_SAFE(self.name, +1)
        started = time.perf_counter()
        status = "success"
        try:
            if self.request.is_eager:
                return self._run_eager_with_retry(*args, **kwargs)
            return self.run(*args, **kwargs)
        except self.autoretry_for as exc:
            status = "retry"
            log.warning("任务遇到瞬时错误，交给 Celery 重试", extra={"task": self.name, "error": str(exc)[:300], "retries": self.request.retries})
            # self.retry 本身会抛 Retry 异常；raise 只是让类型检查/阅读者明确这里不会往下走
            raise self.retry(exc=exc)
        except Exception as exc:
            status = "failure"
            log.error("任务失败", extra={"task": self.name, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            raise
        finally:
            TASK_DURATION.labels(self.name).observe(time.perf_counter() - started)
            TASK_TOTAL.labels(self.name, status).inc()
            TASK_IN_PROGRESS_SAFE(self.name, -1)
            # 还原 contextvar，避免同一线程后续的非任务日志还带着这个 task_id
            task_id_var.reset(token)

    def _run_eager_with_retry(self, *args: Any, **kwargs: Any) -> Any:
        """eager 模式（测试/本地）没有 broker 可以重新投递，就在进程内按同样的退避策略重试。"""
        attempt = 0
        while True:
            try:
                return self.run(*args, **kwargs)
            except self.autoretry_for as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                # 2^(attempt-1) 秒指数退避，封顶 retry_backoff_max，再乘缩放系数（测试设 0 就不等待）
                delay = min(2 ** (attempt - 1), self.retry_backoff_max) * EAGER_BACKOFF_SCALE
                log.warning("eager 模式瞬时错误，进程内重试", extra={"task": self.name, "attempt": attempt, "error": str(exc)[:200]})
                time.sleep(delay)
