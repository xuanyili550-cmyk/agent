"""数据库读写的唯一入口：把 03_STRUCTURED_DATA 的强类型对象存进表、从表里还原回来。

队列任务和 API 都只调这里的函数，不直接拼 ORM 对象——字段怎么落到列上只在这一个文件里决定。
所有函数都接收调用方给的 ``Session``，不自己 commit（由 ``session_scope()`` 统一提交/回滚）。
"""

from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models as m

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _schemas():
    return importlib.import_module("03_STRUCTURED_DATA.schemas")


SCOPE_SEP = "__"


def scoped_id(project_id: str, business_id: str) -> str:
    """业务 id（ep_001 / shot_001_01_02 / char_su_wanwan）在每个项目里都从头编号，
    直接当主键会跨项目互相覆盖。数据库主键统一加项目前缀：``<project_id>__<业务 id>``；
    ``data`` JSON 里保留原始业务 id，读回 03 schema 对象时不受影响。"""
    if business_id.startswith(project_id + SCOPE_SEP):
        return business_id
    return f"{project_id}{SCOPE_SEP}{business_id}"


def unscoped_id(db_id: str) -> str:
    return db_id.split(SCOPE_SEP, 1)[1] if SCOPE_SEP in db_id else db_id


def _upsert(db: Session, orm_cls, pk: str, **values):
    # 参数名不能叫 model：assets 表有一列就叫 model，会和 **values 撞车
    row = db.get(orm_cls, pk)
    if row is None:
        row = orm_cls(**{orm_cls.__mapper__.primary_key[0].name: pk}, **values)
        db.add(row)
    else:
        for key, value in values.items():
            setattr(row, key, value)
    return row


# ---- 故事引擎产出 -> 数据库 ----------------------------------------------------------


def persist_drama_state(db: Session, project_id: str, state: dict[str, Any]) -> dict[str, list[str]]:
    """把 02_STORY_ENGINE ``DramaState``（build_graph 的 invoke 结果）整份写进数据库。

    幂等：同一个业务 id 重复写是更新不是重复插入，流水线重跑不会产生脏数据。
    返回写入的各类 id，方便调用方记进 PipelineRun.result。
    """
    written: dict[str, list[str]] = {"story_bibles": [], "characters": [], "episodes": [], "scenes": [], "shots": [], "prompts": []}

    sid = lambda business_id: scoped_id(project_id, business_id)  # noqa: E731

    bible = state.get("story_bible")
    if bible is not None:
        arcs = [arc.model_dump(mode="json") for arc in state.get("season_arcs", [])] or (
            [state["season_arc"].model_dump(mode="json")] if state.get("season_arc") else []
        )
        _upsert(db, m.StoryBible, sid(bible.id), project_id=project_id, title=bible.title, data=bible.model_dump(mode="json"), season_arcs=arcs)
        written["story_bibles"].append(sid(bible.id))

    for character in state.get("characters", []):
        _upsert(
            db,
            m.Character,
            sid(character.id),
            project_id=project_id,
            name=character.name,
            description=character.backstory,
            role=character.role.value,
            data=character.model_dump(mode="json"),
        )
        written["characters"].append(sid(character.id))

    episodes = state.get("episodes") or ([state["episode"]] if state.get("episode") else [])
    scripts = {s.episode_id: s for s in (state.get("scripts") or ([state["script"]] if state.get("script") else []))}
    editorials = state.get("editorial_by_episode", {})
    for episode in episodes:
        script = scripts.get(episode.id)
        _upsert(
            db,
            m.Episode,
            sid(episode.id),
            project_id=project_id,
            episode_number=episode.episode_number,
            title=episode.title,
            status="scripted",
            season_id=episode.season_id,
            data=episode.model_dump(mode="json"),
            script=script.model_dump(mode="json") if script else None,
            editorial=editorials.get(episode.id),
        )
        written["episodes"].append(sid(episode.id))
        # 重跑同一集时先清掉旧的场次/镜头/提示词，避免上一版分镜残留在新版里
        _delete_episode_children(db, sid(episode.id))

    for scene in state.get("scenes", []):
        _upsert(
            db,
            m.Scene,
            sid(scene.id),
            episode_id=sid(scene.episode_id),
            scene_number=scene.scene_number,
            description=scene.description,
            location=scene.location_id,
            data=scene.model_dump(mode="json"),
        )
        written["scenes"].append(sid(scene.id))

    scene_to_episode = {s.id: s.episode_id for s in state.get("scenes", [])}
    for shot in state.get("shots", []):
        _upsert(
            db,
            m.Shot,
            sid(shot.id),
            scene_id=sid(shot.scene_id),
            episode_id=sid(scene_to_episode.get(shot.scene_id, f"ep_{shot.episode:03d}")),
            shot_number=shot.shot,
            description=shot.action,
            duration_sec=shot.duration,
            data=shot.model_dump(mode="json"),
        )
        written["shots"].append(sid(shot.id))

    for prompt in state.get("image_prompts", []):
        _upsert(db, m.Prompt, sid(prompt.id), shot_id=sid(prompt.shot_id), kind="image", prompt_text=prompt.prompt_text, data=prompt.model_dump(mode="json"))
        written["prompts"].append(sid(prompt.id))
    for prompt in state.get("video_prompts", []):
        _upsert(db, m.Prompt, sid(prompt.id), shot_id=sid(prompt.shot_id), kind="video", prompt_text=prompt.prompt_text, data=prompt.model_dump(mode="json"))
        written["prompts"].append(sid(prompt.id))

    db.flush()
    return written


def _delete_episode_children(db: Session, episode_db_id: str) -> None:
    shot_ids = [r.shot_id for r in db.execute(select(m.Shot.shot_id).where(m.Shot.episode_id == episode_db_id)).all()]
    if shot_ids:
        for prompt in db.execute(select(m.Prompt).where(m.Prompt.shot_id.in_(shot_ids))).scalars().all():
            db.delete(prompt)
        for asset in db.execute(select(m.Asset).where(m.Asset.shot_id.in_(shot_ids))).scalars().all():
            asset.shot_id = None  # 历史素材保留但解除关联
        for shot in db.execute(select(m.Shot).where(m.Shot.shot_id.in_(shot_ids))).scalars().all():
            db.delete(shot)
    for scene in db.execute(select(m.Scene).where(m.Scene.episode_id == episode_db_id)).scalars().all():
        db.delete(scene)
    db.flush()


# ---- 读回强类型对象 -------------------------------------------------------------------


def load_shots_for_episode(db: Session, episode_db_id: str) -> list[Any]:
    """返回 (数据库 id, schemas.Shot) 列表，按场次/镜头号排序。"""
    sch = _schemas()
    rows = db.execute(select(m.Shot).where(m.Shot.episode_id == episode_db_id)).scalars().all()
    shots = [(r.shot_id, sch.Shot.model_validate(r.data)) for r in rows if r.data]
    return sorted(shots, key=lambda pair: (pair[1].scene, pair[1].shot))


def load_image_prompt(db: Session, shot_db_id: str) -> Optional[Any]:
    sch = _schemas()
    row = db.execute(select(m.Prompt).where(m.Prompt.shot_id == shot_db_id, m.Prompt.kind == "image")).scalars().first()
    return sch.ImagePrompt.model_validate(row.data) if row else None


def load_episode(db: Session, episode_id: str) -> Optional[Any]:
    sch = _schemas()
    row = db.get(m.Episode, episode_id)
    return sch.Episode.model_validate(row.data) if row and row.data else None


def load_scenes_for_episode(db: Session, episode_db_id: str) -> list[Any]:
    sch = _schemas()
    rows = db.execute(select(m.Scene).where(m.Scene.episode_id == episode_db_id).order_by(m.Scene.scene_number)).scalars().all()
    return [sch.Scene.model_validate(r.data) for r in rows if r.data]


def load_characters(db: Session, character_ids: Iterable[str]) -> list[m.Character]:
    ids = list(character_ids)
    if not ids:
        return []
    return db.execute(select(m.Character).where(m.Character.character_id.in_(ids))).scalars().all()


# ---- 素材 / QC / 用量 -------------------------------------------------------------------


def persist_asset(
    db: Session,
    record: Any,
    *,
    kind: str = "image",
    qc_status: str = "pending",
    attempt: int = 1,
    retry_level: str | None = None,
    width: int | None = None,
    height: int | None = None,
    storage_key: str | None = None,
) -> m.Asset:
    """07_GENERATION.asset_registry.AssetRecord -> assets 表。外键指向的 shot/episode/character
    不存在时置空而不是报错：手工调用 image_task 时未必先建了 Shot。"""
    shot_id = record.shot_id if record.shot_id and db.get(m.Shot, record.shot_id) else None
    episode_id = record.episode_id if record.episode_id and db.get(m.Episode, record.episode_id) else None
    character_id = record.character_id if record.character_id and db.get(m.Character, record.character_id) else None
    return _upsert(
        db,
        m.Asset,
        record.asset_id,
        file_path=record.file_path,
        kind=kind,
        character_id=character_id,
        episode_id=episode_id,
        shot_id=shot_id,
        model=record.model,
        prompt=record.prompt,
        seed=record.seed,
        version=record.version,
        license=record.license,
        qc_status=qc_status,
        attempt=attempt,
        retry_level=retry_level,
        width=width,
        height=height,
        storage_key=storage_key,
    )


def persist_qc_report(db: Session, report: Any, *, attempt: int = 1) -> m.QCReportRow:
    payload = report.model_dump(mode="json") if hasattr(report, "model_dump") else dict(report)
    row = m.QCReportRow(
        asset_id=payload.get("asset_id") if payload.get("asset_id") and db.get(m.Asset, payload["asset_id"]) else None,
        shot_id=payload.get("shot_id"),
        decision=payload["decision"],
        attempt=attempt,
        data=payload,
    )
    db.add(row)
    db.flush()
    return row


def mark_shot_result(db: Session, shot_id: str, *, status: str, image_path: str | None = None, video_path: str | None = None) -> None:
    row = db.get(m.Shot, shot_id)
    if row is None:
        return
    row.status = status
    if image_path:
        row.image_path = image_path
    if video_path:
        row.video_path = video_path


def record_llm_usage(
    db: Session,
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    context_id: str | None = None,
    run_id: str | None = None,
    agent: str | None = None,
) -> m.LLMUsage:
    row = m.LLMUsage(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        context_id=context_id,
        run_id=run_id,
        agent=agent,
    )
    db.add(row)
    return row


# ---- 流水线运行 -------------------------------------------------------------------------


def update_run(db: Session, run_id: str, *, status: str | None = None, stage: str | None = None, error: str | None = None, **result) -> m.PipelineRun:
    run = db.get(m.PipelineRun, run_id)
    if run is None:
        raise LookupError(f"PipelineRun {run_id} 不存在")
    if status:
        run.status = status
    if stage:
        run.stage = stage
    if error is not None:
        run.error = error
    if result:
        merged = dict(run.result or {})
        merged.update(result)
        run.result = merged
    run.updated_at = datetime.now(UTC)
    return run
