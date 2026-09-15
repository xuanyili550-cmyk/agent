"""Celery 应用装配：broker/backend、任务注册、队列路由、可靠性参数、worker 侧日志与指标。

这是 API 进程和 worker 进程共用的一份配置：API 通过 ``celery_app`` 入队（``task.delay``）和查结果
（``AsyncResult``），worker 用 ``celery -A 13_INFRA.workers.celery_app worker -Q <队列>`` 启动。
任务按资源类型分到 llm / image / video / tts / lipsync / qc 六个队列，这样可以给 GPU 队列和
CPU 队列分别部署不同规格、不同并发数的 worker。
"""

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
    # 所有任务的基类：统一做 task_id 上下文、指标、失败落库等横切逻辑
    task_cls=BaseTask,
    # include 而不是 autodiscover：包名数字开头，autodiscover 找不到；显式列出也更清楚有哪些任务
    include=[
        "13_INFRA.queue.llm_task",
        "13_INFRA.queue.assistant_task",
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
    # 只用 JSON 序列化：pickle 有反序列化任意代码的安全风险，而且 JSON 便于跨语言/排查
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # 任务名 -> 队列：故事/LLM 类走 llm，生图/镜头走 image（GPU），渲染/打包/发布/门控这类轻量编排任务复用 qc 队列
    task_routes={
        "llm_task": {"queue": "llm"},
        "story_task": {"queue": "llm"},
        "assistant_task": {"queue": "llm"},  # 制片助理也要调 LLM，和故事阶段共用 llm worker
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
    # acks_late：任务执行完才向 broker 确认，worker 中途被 OOM/抢占杀掉时消息会回到队列重新投递，而不是丢失
    task_acks_late=True,
    # worker 进程异常退出（不是任务自己抛异常）时把消息退回队列，配合 acks_late 使用
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    # 记录 STARTED 状态，API 查询时能区分"排队中"和"执行中"
    task_track_started=True,
    # 软超时先抛 SoftTimeLimitExceeded 让任务有机会清理，硬超时直接杀进程
    task_soft_time_limit=settings.task_soft_time_limit_seconds,
    task_time_limit=settings.task_time_limit_seconds,
    # 结果保留 7 天：够排查问题，又不会让 Redis 无限膨胀
    result_expires=7 * 24 * 3600,
    # 让 demo/测试和本地开发在没有 Redis 的情况下同步执行任务；生产禁止（config 里会拦）
    task_always_eager=settings.celery_task_always_eager,
    # 不开这个的话 eager 模式的结果不会写进 backend，之后 AsyncResult(task_id) 会永远 PENDING
    task_store_eager_result=True,
    # eager 模式下任务异常直接抛给调用方（API 路由据此判断 run 是否已标 failed）
    task_eager_propagates=True,
)


@setup_logging.connect
def _setup_logging(**_kwargs) -> None:
    """Celery ``setup_logging`` 信号回调：用平台自己的结构化日志配置替换 Celery 默认的。

    连接了这个信号后 Celery 就不再自己配 logging；``force=True`` 是因为模块导入阶段可能已经配置过一次。
    """
    # 接管 Celery 自己的日志配置，worker 日志也走结构化格式
    configure_logging(settings.log_level, settings.log_json, force=True)


@worker_process_init.connect
def _start_metrics_server(**_kwargs) -> None:
    """每个 worker 子进程启动时起一个 Prometheus 指标 HTTP 端口。

    挂在 ``worker_process_init`` 而不是模块顶层：prefork 模式下指标要在子进程里采集才准确，
    而且 API 进程导入本模块时不应该顺带起端口。
    """
    if not settings.metrics_enabled:
        return
    try:
        from prometheus_client import start_http_server

        start_http_server(settings.worker_metrics_port)
    except OSError:
        # 同机多个 worker 进程（prefork）会抢同一个端口：只有第一个能起，其余静默跳过。
        # 生产按容器部署，每个容器一个端口，不会碰到这个问题。
        pass
