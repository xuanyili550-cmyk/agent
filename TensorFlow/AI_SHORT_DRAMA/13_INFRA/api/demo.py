"""AI 短剧 API 的独立 CRUD 演示/冒烟测试，不需要真实 Postgres/Redis：
数据库用 SQLite 内存库，任务用 Celery eager 模式（无 broker，同步执行）。

在整个平台里它相当于"最小可运行验证"：确认 FastAPI 路由、鉴权、分页、SQLAlchemy 模型、
Celery 任务入队与结果回读这条链路在零外部依赖下能跑通；``test_demo.py`` 把它接进 pytest。

直接运行：
    python 13_INFRA/api/demo.py
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

# 必须在 import 13_INFRA.* 之前设好环境变量：config/get_settings 在模块导入时就会读取它们。
# setdefault 而不是直接赋值：调用方（CI / 本地）已经设置的值优先。
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("STORAGE_LOCAL_ROOT", tempfile.mkdtemp(prefix="ai_short_drama_storage_"))
os.environ.setdefault("ARTIFACTS_ROOT", tempfile.mkdtemp(prefix="ai_short_drama_artifacts_"))
os.environ.setdefault("API_KEYS", "demo-key")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("QC_BACKEND", "none")
os.environ.setdefault("METRICS_ENABLED", "false")
os.environ.setdefault("APP_ENV", "test")
# 受保护接口都要带这个头；取 API_KEYS 里第一个 key
DEMO_HEADERS = {"X-API-Key": os.environ["API_KEYS"].split(",")[0]}

# 项目根目录进 sys.path，importlib 才能按 "13_INFRA.api.main" 找到包（数字开头的包名不能用普通 import）
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def run_demo() -> None:
    """装配 TestClient 并跑完整个冒烟流程。

    用 ``importlib.import_module`` 而不是 import 语句：包名 ``13_INFRA`` 以数字开头，语法上不能直接 import。
    ``with TestClient(...)`` 会触发应用的 lifespan（建表等），所以请求必须在 with 块内发。
    """
    from fastapi.testclient import TestClient

    main_mod = importlib.import_module("13_INFRA.api.main")
    models_mod = importlib.import_module("13_INFRA.database.models")
    session_mod = importlib.import_module("13_INFRA.database.session")

    with TestClient(main_mod.app, headers=DEMO_HEADERS) as client:
        _run_requests(client, models_mod, session_mod)


def _run_requests(client, models_mod, session_mod) -> None:
    """依次走：健康检查 -> 鉴权 -> 建 project/character/episode/scene/shot/asset -> 列表分页 -> 任务入队与查状态 -> 删除。

    每一步都用 assert 校验状态码，失败时把响应正文带出来方便定位。
    """
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    print("[health]", resp.json())

    # Web 控制台：不带 key 也能打开页面（页面里的请求才带 key）
    resp = client.get("/", headers={"X-API-Key": ""})
    assert resp.status_code == 200 and "AI 短剧控制台" in resp.text, resp.text[:200]
    print("[console] GET / ->", resp.headers["content-type"], f"{len(resp.text)} bytes")

    # 鉴权：没带 key 一律 401
    resp = client.get("/projects", headers={"X-API-Key": ""})
    assert resp.status_code == 401, resp.text
    print("[auth] missing key -> 401")

    resp = client.post("/projects", json={"name": "Demo Project", "description": "smoke test"})
    assert resp.status_code == 201, resp.text
    project = resp.json()
    print("[create project]", project)

    resp = client.post(
        "/characters",
        json={"project_id": project["project_id"], "name": "Lena", "description": "protagonist"},
    )
    assert resp.status_code == 201, resp.text
    character = resp.json()
    print("[create character]", character)

    resp = client.post(
        "/episodes",
        json={"project_id": project["project_id"], "episode_number": 1, "title": "Pilot"},
    )
    assert resp.status_code == 201, resp.text
    episode = resp.json()
    print("[create episode]", episode)

    # Scene 没有对外挂 API（按规格只挂了 projects/episodes/characters/shots/assets/tasks），
    # 所以直接用数据库 session 建一条，让后面的 Shot 有合法的 scene_id 可引用。
    db = session_mod.SessionLocal()
    try:
        scene = models_mod.Scene(episode_id=episode["episode_id"], scene_number=1, description="Opening scene")
        db.add(scene)
        db.commit()
        db.refresh(scene)
        scene_id = scene.scene_id
    finally:
        db.close()
    print("[create scene] (direct DB)", scene_id)

    resp = client.post(
        "/shots",
        json={"scene_id": scene_id, "shot_number": 1, "description": "Wide establishing shot", "duration_sec": 3.5},
    )
    assert resp.status_code == 201, resp.text
    shot = resp.json()
    print("[create shot]", shot)

    resp = client.post(
        "/assets",
        json={
            "file_path": "/tmp/demo_asset.png",
            "character_id": character["character_id"],
            "episode_id": episode["episode_id"],
            "shot_id": shot["shot_id"],
            "model": "stable-diffusion-xl",
            "prompt": "a cinematic portrait",
            "seed": 42,
            "version": "v1",
            "license": "internal-use",
        },
    )
    assert resp.status_code == 201, resp.text
    asset = resp.json()
    print("[create asset]", asset)

    # 每种资源的列表接口都要能查到刚建的那条，且带 X-Total-Count 分页头
    created = {
        "/projects": project["project_id"],
        "/characters": character["character_id"],
        "/episodes": episode["episode_id"],
        "/shots": shot["shot_id"],
        "/assets": asset["asset_id"],
    }
    for path, wanted in created.items():
        resp = client.get(path, params={"limit": 200, "offset": 0})
        assert resp.status_code == 200, resp.text
        items = resp.json()
        # "/projects" -> "project_id"：去掉斜杠和结尾的 s 再拼 _id，正好是各资源的主键字段名
        key = path.strip("/")[:-1] + "_id"
        assert any(item[key] == wanted for item in items), f"{wanted} not listed in {path}"
        assert int(resp.headers["X-Total-Count"]) >= len(items)
        print(f"[list {path}]", len(items), "item(s), total", resp.headers["X-Total-Count"])

    resp = client.get(f"/assets/{asset['asset_id']}")
    assert resp.status_code == 200, resp.text
    print("[get asset]", resp.json())

    # 任务队列往返：入队一个 image 任务。image_task 现在会真的调
    # 07_GENERATION.image.image_generator.DummyImageGenerator（不需要凭据/GPU），
    # 所以它会成功并返回一个真实的占位图路径。
    # 先发一个带未知字段的 payload：task_payloads 是 extra="forbid"，应在 API 入口就被拦成 422
    resp = client.post("/tasks", json={"queue": "image", "payload": {"prompt": "a red door", "bogus": 1}})
    assert resp.status_code == 422, resp.text
    print("[enqueue task] unknown field -> 422")
    resp = client.post("/tasks", json={"queue": "image", "payload": {"prompt": "a red door", "width": 256, "height": 256}})
    assert resp.status_code == 202, resp.text
    task = resp.json()
    print("[enqueue task]", task)

    # eager 模式下任务在 delay() 里就同步跑完了，所以这里立刻能查到 SUCCESS
    resp = client.get(f"/tasks/{task['task_id']}")
    assert resp.status_code == 200, resp.text
    status = resp.json()
    print("[task status]", status)
    assert status["status"] == "SUCCESS", status
    assert status["result"]["file_path"], status

    resp = client.delete(f"/assets/{asset['asset_id']}")
    assert resp.status_code == 204, resp.text
    resp = client.get(f"/assets/{asset['asset_id']}")
    assert resp.status_code == 404
    print("[delete asset] ok, 404 on refetch")

    # 制片助理（ToolAgent）：一句创意跑完故事阶段（mock LLM）后，让带工具的 Agent 在产出上做检查。
    # 走的是 assistant_task（llm 队列），eager 模式同步返回答案和每一步工具调用记录。
    resp = client.post(
        "/pipelines/episodes",
        json={
            "project_id": project["project_id"],
            "idea": "豪门千金重生复仇",
            "num_characters": 3,
            "num_scenes": 1,
            "prompt_mode": "template",
            "planner": "rule",
        },
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["run_id"]
    resp = client.post(f"/pipelines/{run_id}/assistant", json={"question": "帮我检查第 1 集有没有引用错误"})
    assert resp.status_code == 202, resp.text
    assistant = resp.json()["result"]
    print("[assistant]", [s["tool"] for s in assistant["steps"]], "->", assistant["answer"])
    assert assistant["stopped_reason"] == "final" and all(s["status"] == "success" for s in assistant["steps"]), assistant

    # 四大监控指标汇总：token 消耗 / 工具调用成功率 / 执行时间 / 错误率
    summary = client.get("/metrics/summary").json()
    print(
        "[metrics/summary]",
        f"tokens={summary['token_usage']['total_tokens']}",
        f"tool_success_rate={summary['tool_calls']['success_rate']}",
        f"tasks={sorted(summary['execution_time']['by_task'])}",
        f"error_rate={summary['error_rate']['error_rate']}",
    )
    # 指标是进程级累计值（pytest 里和其他测试共享），所以只断言本次两个工具都记到了成功
    assert summary["tool_calls"]["by_tool"]["run_rule_check"]["success"] >= 1 and summary["tool_calls"]["by_tool"]["list_episodes"]["success"] >= 1, summary

    print("\nDEMO OK: FastAPI CRUD + task enqueue/status round-trip + assistant agent + metrics summary all passed.")


if __name__ == "__main__":
    run_demo()
