"""发布任务：11_PUBLISH 客户端上传 + 状态轮询回写。

- ``publish_task``：对 manifest 里的每个平台调用 upload（settings.publish_dry_run=True 时只校验
  payload 不发请求）；结果写 publish_records 表并回写 manifest 的 publish_status_per_platform。
- ``publish_status_task``：对状态还在 pending/in_review 的记录调 fetch_status 轮询，直到
  published/failed；用 Celery ``countdown`` 自我调度，间隔逐次拉长。
- 定时发布：``scheduled_at`` 传给 apply_async(eta=...)，由 broker 到点投递。

在流水线里的位置：manifest_task -> **publish_task**（run 状态 publishing -> done）。发布结果同时写两处：
数据库 publish_records（API 查询、统计）和 manifest 文件（10_EPISODES 的交付物自带发布状态）。
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
# 状态轮询间隔（秒），逐次拉长：平台审核通常要几分钟到几十分钟，前密后疏既及时又不浪费请求
POLL_SCHEDULE_SECONDS = [30, 60, 120, 300, 600, 1800]


def _clients(platforms: list[str]) -> dict:
    """按平台名取 11_PUBLISH 的发布客户端；不认识的平台名静默跳过（不在 build_clients 结果里）。
    单独抽成函数是为了测试能 monkeypatch 成假客户端。"""
    cli = mod("11_PUBLISH.publish_cli")
    all_clients = cli.build_clients()
    return {p: all_clients[p] for p in platforms if p in all_clients}


def publish_manifest(
    manifest_path: str, *, platforms: Optional[list[str]] = None, dry_run: Optional[bool] = None, scheduled_at: Optional[datetime] = None
) -> list[Dict[str, Any]]:
    """把一份 manifest 发到各平台：逐平台 upload -> 写 PublishRecord -> 更新 manifest 里的状态 -> 保存 manifest。

    platforms / dry_run 不传时用 settings 默认值。每个平台各自 try/except，一个平台的异常记成 FAILED
    记录而不中断其它平台。返回每个平台一条结果 dict（含 publish_id，供后续轮询任务用）。
    """
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
        # 每个平台单独一个事务：一条记录写失败不会连累已写好的其它平台记录
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
            # flush 让 publish_id 默认值生成，出了 with 之后 row 已 expire，所以要在事务内取出来
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
    """链式任务：接 manifest_task 的结果，逐集发布；真实发布（非 dry-run）且状态还没终态的记录
    安排第一次状态轮询。完成后把 run 标成 done——这是整条流水线的最后一环。"""
    with session_scope() as db:
        repo.update_run(db, run_id, status="publishing", stage="publish")
    published = []
    for m in manifest_result.get("manifests", []):
        published.extend(publish_manifest(m["manifest_path"], dry_run=dry_run))
        for record in published:
            # dry-run 没有真正上传，没有 remote_id 可查；已经是终态的也不用轮询
            if record["status"] in ("pending", "in_review") and not record["dry_run"]:
                publish_status_task.apply_async(args=[record["publish_id"], m["manifest_path"], 0], countdown=POLL_SCHEDULE_SECONDS[0])
    with session_scope() as db:
        repo.update_run(db, run_id, status="done", stage="publish_done", publish=published)
    return {"run_id": run_id, "publish": published}


@celery_app.task(name="publish_status_task", bind=True)
def publish_status_task(self, publish_id: str, manifest_path: str, poll_index: int = 0) -> Dict[str, Any]:
    """轮询一条发布记录的平台状态并回写数据库和 manifest；未到终态就按 POLL_SCHEDULE_SECONDS 的下一档
    再调度一次自己，档位用完就停（不无限轮询）。

    记录不存在返回 missing；平台客户端缺失或没有 remote_id（dry-run / 上传失败）返回 unsupported。
    """
    em = mod("10_EPISODES.episode_manifest")
    # 第一个事务只把需要的标量取出来：不把 ORM 行带出 session 去调外部 API
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
    # manifest 文件可能已被人挪走/删除，这种情况只更新数据库，不报错
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
