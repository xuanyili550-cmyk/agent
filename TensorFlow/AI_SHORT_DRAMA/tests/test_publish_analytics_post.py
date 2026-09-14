"""11_PUBLISH 状态轮询、12_ANALYTICS 接入与计算、09_POST 组装渲染。"""

from __future__ import annotations

import importlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

HAS_FFMPEG = subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0


# ---- 11_PUBLISH --------------------------------------------------------------------------


def _manifest(tmp_path) -> Path:
    em = importlib.import_module("10_EPISODES.episode_manifest")
    video = tmp_path / "ep.mp4"
    video.write_bytes(b"\x00" * 1024)
    m = em.EpisodeManifest(
        episode_id="ep_001",
        series_id="s1",
        episode_number=1,
        title="重生夜",
        video_path=str(video),
        duration_sec=12.0,
        shot_ids=["shot_001_01_01"],
        created_at=datetime.now(UTC),
    )
    path = tmp_path / "episode_manifest.json"
    m.save(path)
    return path


def test_publish_dry_run_writes_records_and_manifest(tmp_path):
    publish = importlib.import_module("13_INFRA.queue.publish_task")
    models = importlib.import_module("13_INFRA.database.models")
    session_mod = importlib.import_module("13_INFRA.database.session")
    path = _manifest(tmp_path)
    results = publish.publish_manifest(str(path), platforms=["youtube", "tiktok", "reelshort"], dry_run=True)
    assert {r["platform"] for r in results} == {"youtube", "tiktok", "reelshort"}
    assert all(r["dry_run"] for r in results)
    assert {r["status"] for r in results} <= {"pending", "failed"}
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["publish_status_per_platform"]["youtube"]["status"] == "pending"
    with session_mod.session_scope() as db:
        assert db.query(models.PublishRecord).filter_by(episode_id="ep_001", platform="youtube").count() >= 1


def test_youtube_fetch_status_maps_processing_states():
    cli = importlib.import_module("11_PUBLISH.publish_cli")
    yt = cli._load_client_module("youtube")
    client = yt.YouTubePublishClient()

    class FakeService:
        def __init__(self, upload_status):
            self.upload_status = upload_status

        def videos(self):
            return self

        def list(self, part, id):
            self.vid = id
            return self

        def execute(self):
            return {"items": [{"status": {"uploadStatus": self.upload_status, "rejectionReason": "copyright"}}]}

    client._service = FakeService("processed")
    assert client.fetch_status("abc")[0].value == "published"
    client._service = FakeService("uploaded")
    assert client.fetch_status("abc")[0].value == "in_review"
    client._service = FakeService("rejected")
    status, _, reason = client.fetch_status("abc")
    assert status.value == "rejected" and reason == "copyright"


def test_tiktok_fetch_status(monkeypatch):
    cli = importlib.import_module("11_PUBLISH.publish_cli")
    tk = cli._load_client_module("tiktok")
    client = tk.TikTokPublishClient(access_token="t")

    class Resp:
        def __init__(self, data):
            self._data = data

        def json(self):
            return {"data": self._data}

    monkeypatch.setattr(tk.requests, "post", lambda *a, **k: Resp({"status": "PUBLISH_COMPLETE", "publicaly_available_post_id": [123]}))
    status, url, _ = client.fetch_status("pid")
    assert status.value == "published" and url.endswith("/123")
    monkeypatch.setattr(tk.requests, "post", lambda *a, **k: Resp({"status": "FAILED", "fail_reason": "too long"}))
    assert client.fetch_status("pid")[2] == "too long"


def test_publish_status_task_updates_record_and_manifest(tmp_path, monkeypatch):
    publish = importlib.import_module("13_INFRA.queue.publish_task")
    models = importlib.import_module("13_INFRA.database.models")
    session_mod = importlib.import_module("13_INFRA.database.session")
    em = importlib.import_module("10_EPISODES.episode_manifest")
    path = _manifest(tmp_path)
    with session_mod.session_scope() as db:
        row = models.PublishRecord(episode_id="ep_001", platform="youtube", status="in_review", remote_id="vid1", dry_run=False)
        db.add(row)
        db.flush()
        publish_id = row.publish_id

    class FakeClient:
        def fetch_status(self, remote_id):
            return em.PublishStatus.PUBLISHED, "https://youtu.be/vid1", None

    monkeypatch.setattr(publish, "_clients", lambda platforms: {"youtube": FakeClient()})
    out = publish.publish_status_task.apply(args=[publish_id, str(path), 0]).get()
    assert out["status"] == "published"
    with session_mod.session_scope() as db:
        assert db.get(models.PublishRecord, publish_id).remote_url == "https://youtu.be/vid1"
    assert json.loads(path.read_text())["publish_status_per_platform"]["youtube"]["status"] == "published"


# ---- 12_ANALYTICS -------------------------------------------------------------------------


def test_analytics_ingest_and_snapshot(tmp_path):
    ingest = importlib.import_module("12_ANALYTICS.ingest")
    task = importlib.import_module("13_INFRA.queue.analytics_task")
    models = importlib.import_module("13_INFRA.database.models")
    session_mod = importlib.import_module("13_INFRA.database.session")

    events = [
        {"user_id": "u1", "episode_id": "EP1", "event_type": "app_open", "timestamp": "2026-07-01T00:00:00Z"},
        {"user_id": "u1", "episode_id": "EP1", "event_type": "app_open", "timestamp": "2026-07-02T00:00:00Z"},
        {"user_id": "u2", "episode_id": "EP1", "event_type": "app_open", "timestamp": "2026-07-01T00:00:00Z"},
        {"user_id": "u1", "episode_id": "EP1", "surface": "cover", "event_type": "impression", "timestamp": "2026-07-01T01:00:00Z"},
        {"user_id": "u1", "episode_id": "EP1", "surface": "cover", "event_type": "click", "timestamp": "2026-07-01T01:01:00Z"},
        {"user_id": "u1", "episode_id": "EP1", "event_type": "unlock_purchase", "amount_usd": 1.99, "timestamp": "2026-07-01T01:02:00Z"},
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    source = ingest.build_source({"name": "file", "type": "file", "path": str(path)})
    assert task.ingest_events("file", source)["events"] == 6
    # 重复导入去重（主键相同 -> 冲突）：第二次 0 条新增
    with pytest.raises(Exception):
        task.ingest_events("file", source)
    with session_mod.session_scope() as db:
        assert db.query(models.AnalyticsEvent).count() == 6

    class FakeYT:
        def reports(self):
            return self

        def query(self, **kw):
            return self

        def execute(self):
            return {
                "columnHeaders": [{"name": n} for n in ("day", "video", "views", "likes", "comments", "shares", "estimatedMinutesWatched", "estimatedRevenue")],
                "rows": [["2026-07-01", "vid1", 100, 5, 1, 2, 30.5, 0.42]],
            }

    yt = ingest.YouTubeAnalyticsSource(video_episode_map={"vid1": "EP1"}, service=FakeYT())
    assert task.ingest_events("youtube", yt)["metrics"] == 1
    assert task.ingest_events("youtube", yt)["metrics"] == 1  # 同一天同视频是更新不是新增

    snap = task.compute_snapshots()
    retention = {r["day"]: r["retention_rate"] for r in snap["retention"]}
    assert retention[1] == 0.5  # u1 第二天回来了，u2 没有
    assert snap["revenue"]["total_revenue_usd"] == 1.99 and snap["revenue"]["paying_users"] == 1
    assert snap["platform_metrics"]["views"] == 100
    with session_mod.session_scope() as db:
        assert db.query(models.AnalyticsSnapshot).filter_by(metric="retention").count() >= 1
        assert db.query(models.AnalyticsMetric).count() == 1


def test_tiktok_source_batches_and_maps():
    ingest = importlib.import_module("12_ANALYTICS.ingest")

    class Resp:
        def json(self):
            return {"data": {"videos": [{"id": "v1", "view_count": 10, "like_count": 2}]}, "error": {"code": "ok"}}

    src = ingest.TikTokAnalyticsSource(video_episode_map={"v1": "EP1"}, access_token="t", http_post=lambda url, **kw: Resp())
    items = list(src.fetch())
    assert items[0].episode_id == "EP1" and items[0].views == 10 and items[0].kind == "metric"


# ---- 09_POST ------------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_FFMPEG, reason="需要 ffmpeg")
def test_assemble_renders_stills_with_dialogue(tmp_path):
    assemble = importlib.import_module("09_POST.assemble")
    imgs = []
    for i in range(2):
        p = tmp_path / f"s{i}.png"
        Image.new("RGB", (300, 500), color=(i * 100, 50, 50)).save(p)
        imgs.append(p)
    shots = [
        assemble.ShotMedia(shot_id="a", duration_sec=1.0, image_path=str(imgs[0]), dialogue=["第一句", "第二句"]),
        assemble.ShotMedia(shot_id="b", duration_sec=1.5, image_path=str(imgs[1])),
    ]
    cues = assemble.build_dialogue_cues(shots)
    assert len(cues) == 2 and cues[1]["start_sec"] == 0.5
    final = assemble.render_from_media("ep_t", shots, tmp_path / "out", width=180, height=320)
    assert final.exists() and (tmp_path / "out" / "Episode.srt").exists()
    probe = json.loads(
        subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(final)], capture_output=True, text=True).stdout
    )
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (180, 320)
    assert abs(float(probe["format"]["duration"]) - 2.5) < 0.5
    assert any(s["codec_type"] == "audio" for s in probe["streams"])
