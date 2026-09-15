"""数据库读写的唯一入口：把 03_STRUCTURED_DATA 的强类型对象存进表、从表里还原回来。

队列任务和 API 都只调这里的函数，不直接拼 ORM 对象——字段怎么落到列上只在这一个文件里决定。
所有函数都接收调用方给的 ``Session``，不自己 commit（由 ``session_scope()`` 统一提交/回滚）。

函数分四组：
- 主键处理：``scoped_id`` / ``unscoped_id``（业务 id <-> 带项目前缀的数据库主键）；
- 写入：``persist_drama_state``（故事引擎整份产出落库）、``persist_asset`` / ``persist_qc_report`` / ``record_llm_usage``；
- 读回：``load_*``，用 ``schemas.X.model_validate(row.data)`` 把 JSON 还原成 03 的 pydantic 对象；
- 流水线：``update_run``（合并式更新 PipelineRun.result）。
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

# 03_STRUCTURED_DATA 是数字开头的顶层包，只能通过 importlib 按名字 import；先保证项目根在 sys.path 上
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _schemas():
    """延迟 import 03_STRUCTURED_DATA.schemas：模块顶层不 import，避免 alembic / 测试收集时拉起整个 schema 包。"""
    return importlib.import_module("03_STRUCTURED_DATA.schemas")


SCOPE_SEP = "__"


def scoped_id(project_id: str, business_id: str) -> str:
    """业务 id（ep_001 / shot_001_01_02 / char_su_wanwan）在每个项目里都从头编号，
    直接当主键会跨项目互相覆盖。数据库主键统一加项目前缀：``<project_id>__<业务 id>``；
    ``data`` JSON 里保留原始业务 id，读回 03 schema 对象时不受影响。"""
    # 已经带了本项目前缀的 id 原样返回，保证 scoped_id 幂等（API 传进来的可能已经是数据库主键）
    if business_id.startswith(project_id + SCOPE_SEP):
        return business_id
    return f"{project_id}{SCOPE_SEP}{business_id}"


def unscoped_id(db_id: str) -> str:
    """去掉项目前缀，还原成业务 id；没有前缀（手工创建的 uuid 主键）就原样返回。
    只按第一个 ``__`` 切分，因为业务 id 本身不含 ``__`` 而项目 id 是 uuid。"""
    return db_id.split(SCOPE_SEP, 1)[1] if SCOPE_SEP in db_id else db_id


def _upsert(db: Session, orm_cls, pk: str, **values):
    """按主键"有则更新、无则插入"。所有落库都走它，才能保证流水线重跑是幂等的（不会重复插入）。
    用 ``db.get`` 走 identity map，同一个 session 内重复 upsert 同一行不会多发查询。"""
    # 参数名不能叫 model：assets 表有一列就叫 model，会和 **values 撞车
    row = db.get(orm_cls, pk)
    if row is None:
        # 主键列名各表不同（project_id / shot_id ...），从 mapper 里取，不用每张表各写一个 upsert
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

    写入顺序 StoryBible -> Character -> Episode -> Scene -> Shot -> Prompt 是外键依赖顺序；
    state 里单集（episode/script/season_arc）和多集（episodes/scripts/season_arcs）两种形态都兼容。
    """
    written: dict[str, list[str]] = {"story_bibles": [], "characters": [], "episodes": [], "scenes": [], "shots": [], "prompts": []}

    sid = lambda business_id: scoped_id(project_id, business_id)  # noqa: E731

    bible = state.get("story_bible")
    if bible is not None:
        # 多季规划用 season_arcs，旧的单季流程只有 season_arc；统一成列表存
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
    # 剧本按 episode_id 索引，方便下面每集找到自己的 script
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

    # Shot 只带 scene_id，episode_id 要通过 scene 反查；查不到时按 shot.episode 编号拼出业务 id 兜底
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

    # flush 而不是 commit：让主键冲突之类的错误在这里暴露，提交仍由 session_scope 统一做
    db.flush()
    return written


def _delete_episode_children(db: Session, episode_db_id: str) -> None:
    """删掉一集名下的全部 Scene / Shot / Prompt（重跑该集前清场）。
    素材（Asset）不删只解除 shot_id 关联：生成过的图片和它的 QC 记录是成本凭证，要留着可追溯。
    删除顺序 Prompt -> Shot -> Scene 是外键依赖的逆序。"""
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
    # 先 flush 掉 DELETE，随后同一 session 里 upsert 同名 Scene/Shot 才不会撞主键
    db.flush()


# ---- 读回强类型对象 -------------------------------------------------------------------


def load_shots_for_episode(db: Session, episode_db_id: str) -> list[Any]:
    """返回 (数据库 id, schemas.Shot) 列表，按场次/镜头号排序。"""
    sch = _schemas()
    rows = db.execute(select(m.Shot).where(m.Shot.episode_id == episode_db_id)).scalars().all()
    # 没有 data 的行是 API 手工建的空镜头，没法还原成 schemas.Shot，跳过
    shots = [(r.shot_id, sch.Shot.model_validate(r.data)) for r in rows if r.data]
    return sorted(shots, key=lambda pair: (pair[1].scene, pair[1].shot))


def load_image_prompt(db: Session, shot_db_id: str) -> Optional[Any]:
    """读回某个镜头的 schemas.ImagePrompt（kind="image" 那一行）；没有就返回 None。
    镜头生产时把它传给 PromptAgent 做重写，重写需要完整对象而不只是 prompt_text。"""
    sch = _schemas()
    row = db.execute(select(m.Prompt).where(m.Prompt.shot_id == shot_db_id, m.Prompt.kind == "image")).scalars().first()
    return sch.ImagePrompt.model_validate(row.data) if row else None


def load_episode(db: Session, episode_id: str) -> Optional[Any]:
    """按数据库主键读回 schemas.Episode；行不存在或没有 data（手工建的空集）都返回 None。"""
    sch = _schemas()
    row = db.get(m.Episode, episode_id)
    return sch.Episode.model_validate(row.data) if row and row.data else None


def load_scenes_for_episode(db: Session, episode_db_id: str) -> list[Any]:
    """读回一集的全部 schemas.Scene，按场次号排序（渲染时从场次的对白里取字幕）。"""
    sch = _schemas()
    rows = db.execute(select(m.Scene).where(m.Scene.episode_id == episode_db_id).order_by(m.Scene.scene_number)).scalars().all()
    return [sch.Scene.model_validate(r.data) for r in rows if r.data]


def load_characters(db: Session, character_ids: Iterable[str]) -> list[m.Character]:
    """按主键批量取 Character ORM 行（返回 ORM 行而不是 schema 对象，因为调用方要的是 reference_image_path / lora_path 列）。"""
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
    """08_QC 的 QCReport（pydantic 对象或普通 dict 都行）-> qc_reports 表，每次调用新增一行不覆盖。
    asset_id 指向的素材不存在时置空，和 persist_asset 同一个"不因外键缺失报错"的原则。"""
    payload = report.model_dump(mode="json") if hasattr(report, "model_dump") else dict(report)
    row = m.QCReportRow(
        asset_id=payload.get("asset_id") if payload.get("asset_id") and db.get(m.Asset, payload["asset_id"]) else None,
        shot_id=payload.get("shot_id"),
        decision=payload["decision"],
        attempt=attempt,
        data=payload,
    )
    db.add(row)
    # flush 让 qc_report_id 默认值立刻生成，调用方拿到的 row 可以直接读主键
    db.flush()
    return row


def mark_shot_result(db: Session, shot_id: str, *, status: str, image_path: str | None = None, video_path: str | None = None) -> None:
    """回写镜头生产结果：status 必改，image_path / video_path 只在给了值时覆盖（不会把已有路径清空）。
    Shot 不存在时静默返回：手工调用 shot_task 测试单个镜头时库里可能没有这条记录。"""
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
    """每次 LLM 调用记一行 llm_usage（由 queue._common.make_usage_sink 调用）。
    不 flush：用量记账在主流程之外，尽量少一次往返。"""
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
    """更新 PipelineRun：status / stage 传了才改；``**result`` 合并进 run.result 而不是整个替换，
    这样后续阶段（render / manifest / publish）写自己的键时不会抹掉故事阶段写的 episode_ids 等。
    run 不存在直接抛 LookupError——任务拿着一个不存在的 run_id 继续跑没有意义。"""
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
        # 必须新建一个 dict 再赋值：原地修改 JSON 列的 dict SQLAlchemy 感知不到变更，不会生成 UPDATE
        merged = dict(run.result or {})
        merged.update(result)
        run.result = merged
    # onupdate 只在有列变更时触发；显式写一次保证"只合并 result"也会刷新 updated_at
    run.updated_at = datetime.now(UTC)
    return run
