"""13_INFRA.workers：Celery worker 进程侧的装配（``celery_app`` 应用对象、队列路由、可靠性配置、日志/指标信号）。

本文件不导出任何符号；启动 worker 用 ``celery -A 13_INFRA.workers.celery_app worker``，API 侧则 ``from .workers.celery_app import celery_app``。
"""
