from __future__ import annotations

import sys
from pathlib import Path

from celery import Celery
from celery.signals import setup_logging, worker_process_init

# AI_SHORT_DRAMA 根目录必须在 sys.path 上，队列任务才能通过 importlib 拿到兄弟顶层包
# （例如 "08_QC.reports.qc_report"）——数字开头的包名不能用普通 import 语句，但 importlib 和
# 同包内的相对 import 都没问题。
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ..config import get_settings  # noqa: E402
from ..observability import configure_logging  # noqa: E402
from ..queue.base_task import BaseTask  # noqa: E402

settings = get_settings()

celery_app = Celery(
    "ai_short_drama",
    broker=settings.celery_broker_url,
    backend=settings.result_backend,
    task_cls=BaseTask,
    include=[
        "13_INFRA.queue.llm_task",
        "13_INFRA.queue.image_task",
        "13_INFRA.queue.video_task",
        "13_INFRA.queue.tts_task",
        "13_INFRA.queue.lipsync_task",
        "13_INFRA.queue.qc_task",
        "13_INFRA.queue.shot_task",
        "13_INFRA.queue.story_task",
        "13_INFRA.queue.render_task",
        "13_INFRA.queue.manifest_task",
        "13_INFRA.queue.publish_task",
        "13_INFRA.queue.analytics_task",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "llm_task": {"queue": "llm"},
        "story_task": {"queue": "llm"},
        "image_task": {"queue": "image"},
        "shot_task": {"queue": "image"},
        "video_task": {"queue": "video"},
        "tts_task": {"queue": "tts"},
        "lipsync_task": {"queue": "lipsync"},
        "qc_task": {"queue": "qc"},
        "render_task": {"queue": "qc"},
        "manifest_task": {"queue": "qc"},
        "publish_task": {"queue": "qc"},
        "publish_status_task": {"queue": "qc"},
        "analytics_task": {"queue": "qc"},
        "pipeline_gate_task": {"queue": "qc"},
    },
    # 可靠性：迟确认 + 每次只预取 1 条（GPU 任务耗时长，预取多了会让空闲 worker 干等）
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_soft_time_limit=settings.task_soft_time_limit_seconds,
    task_time_limit=settings.task_time_limit_seconds,
    result_expires=7 * 24 * 3600,
    # 让 demo/测试和本地开发在没有 Redis 的情况下同步执行任务；生产禁止（config 里会拦）
    task_always_eager=settings.celery_task_always_eager,
    # 不开这个的话 eager 模式的结果不会写进 backend，之后 AsyncResult(task_id) 会永远 PENDING
    task_store_eager_result=True,
    task_eager_propagates=True,
)


@setup_logging.connect
def _setup_logging(**_kwargs) -> None:
    # 接管 Celery 自己的日志配置，worker 日志也走结构化格式
    configure_logging(settings.log_level, settings.log_json, force=True)


@worker_process_init.connect
def _start_metrics_server(**_kwargs) -> None:
    if not settings.metrics_enabled:
        return
    try:
        from prometheus_client import start_http_server

        start_http_server(settings.worker_metrics_port)
    except OSError:
        # 同机多个 worker 进程（prefork）会抢同一个端口：只有第一个能起，其余静默跳过。
        # 生产按容器部署，每个容器一个端口，不会碰到这个问题。
        pass
