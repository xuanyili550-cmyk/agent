"""独立 QC 任务：对一个已生成的素材跑 08_QC 全套门禁并落库。

镜头生产（shot_task）内部已经内嵌了 QC + 重试阶梯，不经过这个任务；这里给 API 手工触发 QC、
对视频/音频素材补检、或事后复核用。判定结果同时进 Prometheus（QC_DECISIONS）和 qc_reports 表。
"""

from __future__ import annotations

from typing import Any, Dict

from ..database import repository as repo
from ..database.session import session_scope
from ..observability import QC_DECISIONS
from ..workers.celery_app import celery_app
from ._common import mod

__all__ = ["qc_task"]


@celery_app.task(name="qc_task", bind=True)
def qc_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """QC 队列任务：对一个生成素材跑完整的 08_QC 门禁，返回 QCReport（JSON 可序列化 dict），
    判定 approved/retry，并把报告写进 qc_reports 表。

    payload 与 08_QC.reports.qc_report.run_full_qc() 的关键字参数一致，例如：
      {"asset_id": "...", "shot_id": "...",
       "reference_character_image": "...", "generated_image": "...",
       "scene_description": "...", "scene_media_image": "...",
       "video_path": "...", "audio_path": "...", "attempt": 1}
    """
    qc_report = mod("08_QC.reports.qc_report")
    # attempt 是给落库用的元数据，不是 run_full_qc 的参数，先 pop 掉再把剩余 payload 原样透传
    attempt = int(payload.pop("attempt", 1))
    report = qc_report.run_full_qc(**payload)
    QC_DECISIONS.labels(report.decision.value).inc()
    with session_scope() as db:
        repo.persist_qc_report(db, report, attempt=attempt)
    return report.model_dump(mode="json")
