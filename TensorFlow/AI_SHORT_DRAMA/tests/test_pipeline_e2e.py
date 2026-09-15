"""端到端：创意 -> 故事引擎（mock LLM）-> 编审 -> 等待人工审核 -> 批准 -> 镜头生成（dummy）
-> QC -> 渲染（真实 ffmpeg）-> manifest -> 发布（dry-run）。全程 Celery eager，不需要外部服务。"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

main_mod = importlib.import_module("13_INFRA.api.main")
models = importlib.import_module("13_INFRA.database.models")
session_mod = importlib.import_module("13_INFRA.database.session")
orchestration = importlib.import_module("13_INFRA.orchestration")
config = importlib.import_module("13_INFRA.config")

pytestmark = pytest.mark.skipif(subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0, reason="需要 ffmpeg")


@pytest.fixture(scope="module")
def client(api_headers):
    """整模块共用一个带鉴权头的 TestClient，避免每条用例都重新拉起 FastAPI app。"""
    with TestClient(main_mod.app, headers=api_headers) as c:
        yield c


def _project(client) -> str:
    """建一个测试用项目，返回其 project_id，供各用例复用。"""
    resp = client.post("/projects", json={"name": "e2e", "description": "端到端测试"})
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"]


def test_full_pipeline_with_human_review(client, tmp_root):
    """核心端到端用例：验证"人工审核是硬关卡"这条设计约束——

    故事阶段跑完后流水线必须停在 awaiting_review，只有显式 approve 过的集才会继续生产/渲染/
    发布，未批准的集（ep2）状态原地不动。同时核对数据库里落的是 03 schema 的强类型对象、
    LLM 调用有计费记录、成片分辨率与时长和 manifest/发布记录都符合预期。
    """
    project_id = _project(client)
    resp = client.post(
        "/pipelines/episodes",
        json={
            "project_id": project_id,
            "idea": "豪门千金重生复仇，携手冷峻总裁夺回家族企业",
            "num_characters": 3,
            "num_scenes": 2,
            "episode_numbers": [1, 2],
            "prompt_mode": "llm",
        },
    )
    assert resp.status_code == 202, resp.text
    run = resp.json()
    run_id = run["run_id"]

    # 故事阶段 eager 跑完：require_human_review=True -> 停在 awaiting_review
    run = client.get(f"/pipelines/{run_id}").json()
    assert run["status"] == "awaiting_review", run
    ep1, ep2 = f"{project_id}__ep_001", f"{project_id}__ep_002"
    assert run["result"]["episode_ids"] == [ep1, ep2]
    assert [e["business_id"] for e in run["episodes"]] == ["ep_001", "ep_002"]
    assert run["result"]["shots_total"] == 8
    assert run["result"]["editorial"][ep1]["status"] == "approved"
    assert {e["review_status"] for e in run["episodes"]} == {"pending_review"}

    # 剧本 + 编审详情可查
    script = client.get(f"/episodes/{ep1}/script").json()
    assert script["script"]["content"].startswith("【场景一")
    assert script["editorial"]["qc"]["score"] == 86

    # 数据库里已经有 03 schema 的强类型对象
    with session_mod.session_scope() as db:
        assert db.get(models.StoryBible, f"{project_id}__story_001").data["title"] == "重生之豪门逆袭"
        assert db.get(models.Character, f"{project_id}__char_su_wanwan").data["role"] == "protagonist"
        assert db.get(models.Shot, f"{project_id}__shot_002_01_02").episode_id == ep2
        assert db.get(models.Shot, f"{project_id}__shot_002_01_02").data["id"] == "shot_002_01_02"
        assert db.query(models.Prompt).filter(models.Prompt.prompt_id.like(f"{project_id}__%"), models.Prompt.kind == "image").count() == 8
        usage = db.query(models.LLMUsage).filter_by(run_id=run_id).count()
        assert usage >= 40  # 每次 LLM 调用一行

    # 只批准第 1 集 -> 生产 -> 渲染 -> manifest -> 发布（dry-run），eager 一路跑完
    resp = client.post(f"/pipelines/{run_id}/approve", json={"episode_ids": [ep1], "notes": "第一集可以投产"})
    assert resp.status_code == 200, resp.text
    run = client.get(f"/pipelines/{run_id}").json()
    assert run["status"] == "done", run
    assert run["result"]["approved_shots"] and not run["result"]["failed_shots"]
    assert len(run["result"]["renders"]) == 1
    render = run["result"]["renders"][0]
    assert render["episode_id"] == ep1 and render["rendered_shots"] == ["shot_001_01_01", "shot_001_01_02", "shot_001_02_01", "shot_001_02_02"]
    video = Path(render["video_path"])
    assert video.exists() and video.stat().st_size > 0

    # ffprobe 核实成片：竖屏 256x448、时长约等于 4 个镜头之和
    probe = json.loads(
        subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(video)], capture_output=True, text=True).stdout
    )
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (256, 448)
    assert abs(float(probe["format"]["duration"]) - render["duration_sec"]) < 1.0

    manifest_path = Path(run["result"]["manifests"][0]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["episode_id"] == "ep_001" and manifest["shot_ids"] == render["rendered_shots"]
    assert set(manifest["publish_status_per_platform"]) == {"youtube", "tiktok"}
    assert all(p["dry_run"] and p["status"] == "pending" for p in run["result"]["publish"])

    with session_mod.session_scope() as db:
        ep = db.get(models.Episode, ep1)
        assert ep.status == "rendered" and ep.review_status == "approved" and ep.video_path == str(video)
        assert db.get(models.Episode, ep2).review_status == "pending_review"
        assert db.query(models.Asset).filter_by(qc_status="approved").count() >= 4
        assert db.query(models.QCReportRow).count() >= 4
        assert db.query(models.PublishRecord).filter_by(episode_id="ep_001").count() == 2

    usage = client.get(f"/pipelines/{run_id}/usage").json()
    assert usage["calls"] >= 40 and "StoryAgent" in usage["by_agent"]


def test_pipeline_without_human_review_runs_straight_through(client, monkeypatch):
    """验证 require_human_review=False 时流水线一路跑到 done，不在 awaiting_review 停留；
    同时用 editorial=False 走"跳过编审"分支，确认 editorial 结果标成 skipped 而不是伪造一个通过结果。"""
    settings = config.get_settings()
    monkeypatch.setattr(settings, "require_human_review", False)
    project_id = _project(client)
    resp = client.post(
        "/pipelines/episodes",
        json={
            "project_id": project_id,
            "idea": "重生复仇",
            "num_characters": 3,
            "num_scenes": 1,
            "episode_numbers": [1],
            "prompt_mode": "template",
            "planner": "rule",
            "editorial": False,
        },
    )
    assert resp.status_code == 202, resp.text
    run = client.get(f"/pipelines/{resp.json()['run_id']}").json()
    assert run["status"] == "done", run
    assert list(run["result"]["editorial"].values()) == [{"status": "skipped", "rounds": 0}]
    assert len(run["result"]["renders"][0]["rendered_shots"]) == 2


def test_reject_and_missing_project(client):
    """覆盖几条边界路径：reject 后流水线状态置为 rejected；project_id/run_id 不存在时接口返回 404
    而不是抛异常；按 status 过滤 /pipelines 列表能查到被拒的运行。"""
    project_id = _project(client)
    resp = client.post("/pipelines/episodes", json={"project_id": project_id, "idea": "重生复仇", "num_characters": 3, "num_scenes": 1})
    run_id = resp.json()["run_id"]
    resp = client.post(f"/pipelines/{run_id}/reject", json={"notes": "方向不对"})
    assert resp.status_code == 200 and resp.json()["status"] == "rejected"
    assert client.post("/pipelines/episodes", json={"project_id": "nope", "idea": "重生复仇"}).status_code == 404
    assert client.get("/pipelines/nope").status_code == 404
    listed = client.get("/pipelines", params={"status": "rejected"})
    assert listed.status_code == 200 and any(r["run_id"] == run_id for r in listed.json())
