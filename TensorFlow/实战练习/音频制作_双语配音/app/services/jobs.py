"""
================================================================================
 服务层 · 后台任务管理（异步 job + 轮询进度）
================================================================================
 【为什么需要】视频导出是"慢+CPU 密集"操作(要逐句合成音频、渲帧、跑 ffmpeg)。如果做成同步接口
   (请求进来一直等到 mp4 出来才返回)，会：① 请求超时 ② 一个大导出把线程占死 ③ 前端没法显示进度。
   所以改成【异步任务】：提交后立刻返回 job_id，前端拿着 id 轮询进度，好了再下载。

 【怎么实现(单机版)】
   · 线程池 ThreadPoolExecutor 限制并发(max_workers)——同时最多跑 N 个导出，保护 CPU/内存不被打爆。
   · 内存字典 _jobs 登记每个任务的状态(queued→running→done/error)和进度(0~1)。
   · 进度靠回调：真正干活的函数通过 progress_cb(done,total) 上报，这里换算成 0~1 写进状态。
   · 加锁 _lock：任务在工作线程里改状态、HTTP 线程来读状态，多线程并发访问字典要加锁防竞态。
   · 过期清理 cleanup：完成的任务留一段时间供下载，之后清掉，避免字典无限膨胀。

 【生产演进】单机内存版够小规模用；多机/要持久化就换 Celery/RQ + Redis(把队列和状态外置)。
================================================================================
"""
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from ..core.config import get_settings
from ..core.logging import get_logger

log = get_logger("jobs")
_s = get_settings()
_pool = ThreadPoolExecutor(max_workers=_s.max_concurrent_jobs)   # 并发上限=保护机器
_jobs: dict[str, dict] = {}                                      # job_id → 状态
_lock = threading.Lock()                                         # 保护 _jobs 的并发读写


def _now():
    return time.time()


def submit(fn, *args, **kwargs) -> str:
    """提交一个任务。约定 fn(progress_cb, *args) —— 干活时用 progress_cb(done,total) 上报进度。
    立刻返回 job_id(不等任务完成)。"""
    job_id = uuid.uuid4().hex[:16]
    with _lock:
        _jobs[job_id] = {"status": "queued", "progress": 0.0, "result": None,
                         "error": None, "created": _now()}

    def _progress(done, total):
        """工作线程回调：把"第几/共几"换算成 0~1 进度写进状态(供前端进度条)。"""
        with _lock:
            if job_id in _jobs:
                _jobs[job_id]["progress"] = round(done / max(1, total), 3)
                _jobs[job_id]["status"] = "running"

    def _run():
        """真正在工作线程里执行；捕获所有异常写进状态(绝不让线程静默崩掉)。"""
        try:
            result = fn(_progress, *args, **kwargs)
            with _lock:
                _jobs[job_id].update(status="done", progress=1.0, result=result)
        except Exception as e:  # noqa: BLE001 任何失败都要落到 error 状态，前端才好提示
            log.exception("job %s failed", job_id)
            with _lock:
                _jobs[job_id].update(status="error", error=str(e))

    _pool.submit(_run)          # 丢进线程池(排队/并发受 max_workers 限制)
    return job_id


def get(job_id: str) -> dict | None:
    """查任务状态(返回副本，避免调用方拿到内部字典引用被误改)。"""
    with _lock:
        j = _jobs.get(job_id)
        return dict(j) if j else None


def cleanup():
    """清掉超过保留期的旧任务记录(防 _jobs 无限增长)。"""
    ttl = get_settings().job_retention_seconds
    now = _now()
    with _lock:
        for jid in [k for k, v in _jobs.items() if now - v["created"] > ttl]:
            _jobs.pop(jid, None)
