"""Standalone CRUD demo for the AI Short Drama API, runnable with no real
Postgres/Redis: SQLite in-memory for the DB, Celery eager mode (no broker) for tasks.

Run directly:
    python 13_INFRA/api/demo.py
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("STORAGE_LOCAL_ROOT", tempfile.mkdtemp(prefix="ai_short_drama_storage_"))

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def run_demo() -> None:
    from fastapi.testclient import TestClient

    main_mod = importlib.import_module("13_INFRA.api.main")
    models_mod = importlib.import_module("13_INFRA.database.models")
    session_mod = importlib.import_module("13_INFRA.database.session")

    with TestClient(main_mod.app) as client:
        _run_requests(client, models_mod, session_mod)


def _run_requests(client, models_mod, session_mod) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    print("[health]", resp.json())

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

    # Scene isn't a listed API resource (only projects/episodes/characters/shots/
    # assets/tasks are mounted per spec); create it directly via the DB session so a
    # Shot has a valid scene_id to reference.
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

    for path in ("/projects", "/characters", "/episodes", "/shots", "/assets"):
        resp = client.get(path)
        assert resp.status_code == 200, resp.text
        items = resp.json()
        assert len(items) == 1, f"expected 1 item in {path}, got {items}"
        print(f"[list {path}]", len(items), "item(s)")

    resp = client.get(f"/assets/{asset['asset_id']}")
    assert resp.status_code == 200, resp.text
    print("[get asset]", resp.json())

    # Task queue round-trip: enqueue an image task. image_task now calls the real
    # 07_GENERATION.image.image_generator.DummyImageGenerator (no credentials/GPU
    # needed), so it succeeds and returns a real placeholder file path.
    resp = client.post("/tasks", json={"queue": "image", "payload": {"prompt": "a red door"}})
    assert resp.status_code == 202, resp.text
    task = resp.json()
    print("[enqueue task]", task)

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

    print("\nDEMO OK: FastAPI CRUD + task enqueue/status round-trip all passed.")


if __name__ == "__main__":
    run_demo()
