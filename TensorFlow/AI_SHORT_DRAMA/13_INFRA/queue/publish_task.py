"""发布任务：11_PUBLISH 客户端上传 + 状态轮询回写。

- ``publish_task``：对 manifest 里的每个平台调用 upload（settings.publish_dry_run=True 时只校验
  payload 不发请求）；结果写 publish_records 表并回写 manifest 的 publish_status_per_platform。
- ``publish_status_task``：对状态还在 pending/in_review 的记录调 fetch_status 轮询，直到
  published/failed；用 Celery ``countdown`` 自我调度，间隔逐次拉长。
- 定时发布：``scheduled_at`` 传给 apply_async(eta=...)，由 broker 到点投递。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import get_settings
from ..database import repository as repo
from ..database.models import PublishRecord
from ..database.session import session_scope
from ..observability import get_logger
from ..workers.celery_app import celery_app
from ._common import mod

__all__ = ["publish_task", "publish_status_task", "publish_manifest"]

log = get_logger("queue.publish")
POLL_SCHEDULE_SECONDS = [30, 60, 120, 300, 600, 1800]


def _clients(platforms: list[str]) -> dict:
    cli = mod("11_PUBLISH.publish_cli")
    all_clients = cli.build_clients()
    return {p: all_clients[p] for p in platforms if p in all_clients}


def publish_manifest(
    manifest_path: str, *, platforms: Optional[list[str]] = None, dry_run: Optional[bool] = None, scheduled_at: Optional[datetime] = None
) -> list[Dict[str, Any]]:
    settings = get_settings()
    em = mod("10_EPISODES.episode_manifest")
    manifest = em.EpisodeManifest.load(manifest_path)
    platforms = platforms or settings.publish_platforms
    dry_run = settings.publish_dry_run if dry_run is None else dry_run
    results: list[Dict[str, Any]] = []
    now = datetime.now(UTC)

    for name, client in _clients(platforms).items():
        try:
            result = client.upload(manifest, dry_run=dry_run)
            status, remote_id, remote_url, error = result.status, result.remote_id, result.remote_url, result.error_message
        except Exception as exc:  # 单个平台失败不影响其它平台
            status, remote_id, remote_url, error = em.PublishStatus.FAILED, None, None, f"{type(exc).__name__}: {str(exc)[:300]}"
        manifest.publish_status_per_platform[name] = em.PlatformPublishStatus(
            platform=name, status=status, remote_id=remote_id, remote_url=remote_url, last_attempt_at=now, error_message=error
        )
        with session_scope() as db:
            row = PublishRecord(
                episode_id=manifest.episode_id,
                platform=name,
                status=status.value,
                remote_id=remote_id,
                remote_url=remote_url,
                dry_run=dry_run,
                error_message=error,
                scheduled_at=scheduled_at,
                attempted_at=now,
            )
            db.add(row)
            db.flush()
            results.append(
                {
                    "publish_id": row.publish_id,
                    "platform": name,
                    "status": status.value,
                    "remote_id": remote_id,
                    "remote_url": remote_url,
                    "dry_run": dry_run,
                    "error": error,
                }
            )
    manifest.updated_at = now
    manifest.save(manifest_path)
    return results


@celery_app.task(name="publish_task", bind=True)
def publish_task(self, manifest_result: Dict[str, Any], run_id: str, *, dry_run: Optional[bool] = None) -> Dict[str, Any]:
    with session_scope() as db:
        repo.update_run(db, run_id, status="publishing", stage="publish")
    published = []
    for m in manifest_result.get("manifests", []):
        published.extend(publish_manifest(m["manifest_path"], dry_run=dry_run))
        for record in published:
            if record["status"] in ("pending", "in_review") and not record["dry_run"]:
                publish_status_task.apply_async(args=[record["publish_id"], m["manifest_path"], 0], countdown=POLL_SCHEDULE_SECONDS[0])
    with session_scope() as db:
        repo.update_run(db, run_id, status="done", stage="publish_done", publish=published)
    return {"run_id": run_id, "publish": published}


@celery_app.task(name="publish_status_task", bind=True)
def publish_status_task(self, publish_id: str, manifest_path: str, poll_index: int = 0) -> Dict[str, Any]:
    em = mod("10_EPISODES.episode_manifest")
    with session_scope() as db:
        row = db.get(PublishRecord, publish_id)
        if row is None:
            return {"publish_id": publish_id, "status": "missing"}
        platform, remote_id = row.platform, row.remote_id
    client = _clients([platform]).get(platform)
    if client is None or not remote_id:
        return {"publish_id": publish_id, "status": "unsupported"}
    status, remote_url, error = client.fetch_status(remote_id)
    with session_scope() as db:
        row = db.get(PublishRecord, publish_id)
        row.status = status.value
        row.remote_url = remote_url or row.remote_url
        row.error_message = error
        row.attempted_at = datetime.now(UTC)
    if Path(manifest_path).exists():
        manifest = em.EpisodeManifest.load(manifest_path)
        entry = manifest.publish_status_per_platform.get(platform) or em.PlatformPublishStatus(platform=platform)
        manifest.publish_status_per_platform[platform] = entry.model_copy(
            update={"status": status, "remote_url": remote_url or entry.remote_url, "error_message": error, "last_attempt_at": datetime.now(UTC)}
        )
        manifest.save(manifest_path)
    if status in (em.PublishStatus.PENDING, em.PublishStatus.IN_REVIEW) and poll_index + 1 < len(POLL_SCHEDULE_SECONDS):
        publish_status_task.apply_async(args=[publish_id, manifest_path, poll_index + 1], countdown=POLL_SCHEDULE_SECONDS[poll_index + 1])
    return {"publish_id": publish_id, "status": status.value, "remote_url": remote_url}
