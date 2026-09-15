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
    """造一份最小可用的 EpisodeManifest 并落盘，返回 manifest 文件路径，给发布相关用例复用。"""
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
    """验证 dry_run=True 时仍然会给每个平台写 PublishRecord、回写 manifest 里的发布状态——
    dry-run 只是不真的调用平台 API，落库和 manifest 更新这两件事不能跳过。"""
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
    """验证 YouTube 客户端把 Data API 的 uploadStatus 正确映射成平台无关的 PublishStatus 枚举，
    并且能把拒绝原因透传出来，供上层记录到 PublishRecord.error_message。"""
    cli = importlib.import_module("11_PUBLISH.publish_cli")
    yt = cli._load_client_module("youtube")
    client = yt.YouTubePublishClient()

    class FakeService:
        """假的 YouTube Data API service：只实现 fetch_status 用到的 videos().list().execute() 链路。"""

        def __init__(self, upload_status):
            """记住这次要返回的 uploadStatus，供 execute() 组装响应体。"""
            self.upload_status = upload_status

        def videos(self):
            """真实 API 里 videos() 返回一个可继续链式调用的资源对象，这里直接返回自身。"""
            return self

        def list(self, part, id):
            """记录本次查询的视频 id，方便断言/调试；返回自身以支持继续链式调用 execute()。"""
            self.vid = id
            return self

        def execute(self):
            """返回和真实 API 结构一致的最小响应：一个带 uploadStatus/rejectionReason 的 items 列表。"""
            return {"items": [{"status": {"uploadStatus": self.upload_status, "rejectionReason": "copyright"}}]}

    client._service = FakeService("processed")
    assert client.fetch_status("abc")[0].value == "published"
    client._service = FakeService("uploaded")
    assert client.fetch_status("abc")[0].value == "in_review"
    client._service = FakeService("rejected")
    status, _, reason = client.fetch_status("abc")
    assert status.value == "rejected" and reason == "copyright"


def test_tiktok_fetch_status(monkeypatch):
    """验证 TikTok 客户端把开放平台返回的 status 字符串映射成统一的 PublishStatus，
    并能取出成片链接和失败原因。"""
    cli = importlib.import_module("11_PUBLISH.publish_cli")
    tk = cli._load_client_module("tiktok")
    client = tk.TikTokPublishClient(access_token="t")

    class Resp:
        """假的 requests.Response：只需要 .json()，字段结构照抄 TikTok 开放平台的返回格式。"""

        def __init__(self, data):
            """把要返回的业务数据存起来，包一层 {"data": ...} 是 TikTok 接口的统一信封格式。"""
            self._data = data

        def json(self):
            """模拟 requests.Response.json()。"""
            return {"data": self._data}

    monkeypatch.setattr(tk.requests, "post", lambda *a, **k: Resp({"status": "PUBLISH_COMPLETE", "publicaly_available_post_id": [123]}))
    status, url, _ = client.fetch_status("pid")
    assert status.value == "published" and url.endswith("/123")
    monkeypatch.setattr(tk.requests, "post", lambda *a, **k: Resp({"status": "FAILED", "fail_reason": "too long"}))
    assert client.fetch_status("pid")[2] == "too long"


def test_publish_status_task_updates_record_and_manifest(tmp_path, monkeypatch):
    """验证 publish_status_task 轮询到"已发布"之后，会同时更新数据库 PublishRecord 的
    remote_url 和 manifest 文件里的状态——两处状态必须保持一致，下游发布看板才不会读到过期数据。"""
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
        """假的平台发布客户端：固定返回"已发布"，跳过真实网络调用。"""

        def fetch_status(self, remote_id):
            """直接返回已发布状态和观看链接，不关心传入的 remote_id。"""
            return em.PublishStatus.PUBLISHED, "https://youtu.be/vid1", None

    monkeypatch.setattr(publish, "_clients", lambda platforms: {"youtube": FakeClient()})
    out = publish.publish_status_task.apply(args=[publish_id, str(path), 0]).get()
    assert out["status"] == "published"
    with session_mod.session_scope() as db:
        assert db.get(models.PublishRecord, publish_id).remote_url == "https://youtu.be/vid1"
    assert json.loads(path.read_text())["publish_status_per_platform"]["youtube"]["status"] == "published"


# ---- 12_ANALYTICS -------------------------------------------------------------------------


def test_analytics_ingest_and_snapshot(tmp_path):
    """跑一遍 12_ANALYTICS 的完整链路：文件源导入事件（验证去重主键生效）、YouTube 指标源导入
    （验证"同一天同一视频"是更新不是新增）、最后计算留存/收入/平台指标快照并核对结果数值。"""
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
        """假的 YouTube Analytics Reports API：固定返回一行按天聚合的指标数据。"""

        def reports(self):
            """真实 API 里 reports() 返回可继续链式调用的资源对象，这里直接返回自身。"""
            return self

        def query(self, **kw):
            """接受任意查询参数（时间范围、维度等），返回自身以支持继续链式调用 execute()。"""
            return self

        def execute(self):
            """返回和真实 API 结构一致的最小响应：列头 + 一行数据。"""
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
    """验证 TikTok 指标源能把开放平台的视频列表响应，转换成统一的指标事件并按 video_episode_map
    映射回内部 episode_id。"""
    ingest = importlib.import_module("12_ANALYTICS.ingest")

    class Resp:
        """假的 requests.Response，只需要 .json()。"""

        def json(self):
            """返回一条 TikTok 视频指标 + 一个占位的 error 字段（接口约定的成功标记）。"""
            return {"data": {"videos": [{"id": "v1", "view_count": 10, "like_count": 2}]}, "error": {"code": "ok"}}

    src = ingest.TikTokAnalyticsSource(video_episode_map={"v1": "EP1"}, access_token="t", http_post=lambda url, **kw: Resp())
    items = list(src.fetch())
    assert items[0].episode_id == "EP1" and items[0].views == 10 and items[0].kind == "metric"


# ---- 09_POST ------------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_FFMPEG, reason="需要 ffmpeg")
def test_assemble_renders_stills_with_dialogue(tmp_path):
    """验证 09_POST.assemble 能把静态关键帧按镜头时长拼成视频、字幕时间轴与镜头顺序对齐，
    且最终成片的分辨率、总时长、音轨都符合预期（真实调用 ffmpeg/ffprobe）。"""
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
