from __future__ import annotations

import os
import sys
from pathlib import Path

from celery import Celery

# AI_SHORT_DRAMA root must be on sys.path for queue tasks to reach sibling top-level
# packages (e.g. "08_QC.reports.qc_report") via importlib, since names starting with a
# digit can't be used in a plain `import` statement but do work with importlib and
# with relative imports inside the same package.
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Local dev default; production must set CELERY_BROKER_URL / CELERY_RESULT_BACKEND to
# the real Redis instance (see 13_INFRA/docker/docker-compose.yml).
BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", BROKER_URL)

celery_app = Celery(
    "ai_short_drama",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=[
        "13_INFRA.queue.llm_task",
        "13_INFRA.queue.image_task",
        "13_INFRA.queue.video_task",
        "13_INFRA.queue.tts_task",
        "13_INFRA.queue.lipsync_task",
        "13_INFRA.queue.qc_task",
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
        "image_task": {"queue": "image"},
        "video_task": {"queue": "video"},
        "tts_task": {"queue": "tts"},
        "lipsync_task": {"queue": "lipsync"},
        "qc_task": {"queue": "qc"},
    },
    # Lets the demo/test suite and local dev run tasks synchronously without a
    # running Redis broker; production must NOT set this env var.
    task_always_eager=os.environ.get("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true",
    # Without this, eager-mode results aren't written to the backend, so a later
    # AsyncResult(task_id) lookup (as done by GET /tasks/{id}) would see PENDING
    # forever instead of the real outcome.
    task_store_eager_result=True,
)
