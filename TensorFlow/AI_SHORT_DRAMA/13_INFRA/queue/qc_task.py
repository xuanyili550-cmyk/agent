from __future__ import annotations

import importlib
from typing import Any, Dict

from ..workers.celery_app import celery_app

__all__ = ["qc_task"]


@celery_app.task(name="qc_task", bind=True)
def qc_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """QC queue task: runs the full 08_QC gate on a generated asset and returns a
    QCReport as a JSON-serializable dict, deciding approved/retry.

    payload matches 08_QC.reports.qc_report.run_full_qc()'s keyword args, e.g.:
      {"asset_id": "...", "shot_id": "...",
       "reference_character_image": "...", "generated_image": "...",
       "scene_description": "...", "scene_media_image": "...",
       "video_path": "...", "audio_path": "..."}
    Unlike the other queue/*.py stubs, this one is a real, working integration with
    our own 08_QC module (not a TODO placeholder).
    """
    qc_report = importlib.import_module("08_QC.reports.qc_report")
    report = qc_report.run_full_qc(**payload)
    return report.model_dump(mode="json")
