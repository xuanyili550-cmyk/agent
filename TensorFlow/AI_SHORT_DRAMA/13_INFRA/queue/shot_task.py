"""镜头生成任务：图像生成 -> QC -> 三级重试阶梯 -> 落库。这是生产队列里最核心的一个任务。

流程（每个镜头一个任务，chord 并行）：
    1. 解析角色一致性素材：reference_character_ids -> 数据库 characters.reference_image_path /
       lora_path，查不到再看 06_MODELS/character_loras.json；
    2. 按 RetryLadder 尝试生成：
         初次 -> 失败/QC 不过 -> 同参数换 seed 再试 -> AI 改写提示词再试 -> 降分辨率再试
       每次尝试都写一条 Asset（qc_status 记录该次结果）+ 一条 QCReportRow，可追溯；
    3. 通过：Shot.status=approved、image_path 指向通过的关键帧；阶梯用尽：Shot.status=failed，
       返回 status=failed 但**不抛异常**——chord 里一个镜头失败不该让整集其它镜头白做，
       由 render_task 决定用通过的镜头出片并把失败镜头记进 PipelineRun.result。

QC 后端：settings.qc_backend="clip" 用 08_QC 的 CLIP 角色一致性打分（需要参考图）；
"none" 跳过打分直接 approved（开发/无权重环境）。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict

from ..config import Settings, get_settings
from ..database import repository as repo
from ..database.models import Character
from ..database.session import session_scope
from ..observability import QC_DECISIONS, RETRY_LADDER, get_logger
from ..workers.celery_app import celery_app
from ._common import build_llm_provider, mod

__all__ = ["shot_task", "produce_shot", "resolve_character_assets"]

log = get_logger("queue.shot")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---- 角色一致性素材 ---------------------------------------------------------------------


def resolve_character_assets(character_ids: list[str], db=None, project_id: str | None = None) -> dict[str, Any]:
    """返回 {"reference_images": [...], "lora_path": str | None, "trigger_words": [...]}。

    优先级：数据库 characters 表 > 06_MODELS/character_loras.json。多个角色同框时只取第一个
    角色的 LoRA（一次只能挂一个），参考图则全部带上。
    """
    refs: list[str] = []
    lora: str | None = None
    triggers: list[str] = []
    registry: dict[str, Any] = {}
    registry_path = _PROJECT_ROOT / "06_MODELS" / "character_loras.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8")).get("characters", {})

    for cid in character_ids:
        row = None
        if db is not None:
            # 数据库主键带项目前缀（见 repository.scoped_id）；手工建的角色可能没有前缀，两种都查
            row = (db.get(Character, repo.scoped_id(project_id, cid)) if project_id else None) or db.get(Character, cid)
        entry = registry.get(cid, {})
        ref = row.reference_image_path if row and row.reference_image_path else None
        if ref and Path(ref).exists():
            refs.append(ref)
        else:
            for candidate in entry.get("reference_images", []):
                path = _PROJECT_ROOT / candidate if not Path(candidate).is_absolute() else Path(candidate)
                if path.exists():
                    refs.append(str(path))
        if lora is None:
            lora_candidate = (row.lora_path if row and row.lora_path else None) or entry.get("lora_path")
            if lora_candidate and Path(lora_candidate).exists():
                lora = lora_candidate
        if entry.get("trigger_word"):
            triggers.append(entry["trigger_word"])
    return {"reference_images": refs, "lora_path": lora, "trigger_words": triggers}


# ---- QC ------------------------------------------------------------------------------


def score_asset(settings: Settings, *, generated_image: str, reference_images: list[str], shot_id: str, asset_id: str, character_id: str | None):
    """返回 08_QC 的 QCReport。clip 后端需要至少一张参考图，否则只做"文件存在"级检查。"""
    qc = mod("08_QC.reports.qc_report")
    thresholds = qc.QCThresholds(character_min_similarity=settings.qc_character_min_similarity, video_required=False, audio_required=False)
    if settings.qc_backend == "clip" and reference_images:
        checker_mod = mod("08_QC.character.consistency_checker")
        checker = checker_mod.CharacterConsistencyChecker()
        result = checker.score_against_references(reference_images, generated_image, character_id=character_id)
        return qc.build_qc_report(asset_id=asset_id, shot_id=shot_id, character_result=result, thresholds=thresholds)
    return qc.build_qc_report(asset_id=asset_id, shot_id=shot_id, thresholds=thresholds)


# ---- 提示词改写（三级重试第二级） ------------------------------------------------------


def rule_based_rewrite(prompt: str, reasons: list[str]) -> str:
    """没有 LLM 时的兜底改写：去掉常见触发安全过滤的词，补上画质/清晰度描述。"""
    risky = ["血", "blood", "gore", "尸体", "corpse", "裸", "nude", "暴力", "violence"]
    cleaned = prompt
    for word in risky:
        cleaned = cleaned.replace(word, "")
    suffix = "，人物面部清晰完整，五官自然，构图居中，高清细节，电影级布光"
    return cleaned.strip("，, ") + suffix


def make_prompt_rewriter(settings: Settings, image_prompt_data: dict[str, Any] | None) -> Callable[[str, list[str], int], str]:
    """返回 (prompt, reasons, attempt) -> new_prompt。有 LLM 就用 PromptAgent，没有就规则改写。"""
    if settings.llm_provider == "mock" or image_prompt_data is None:
        return lambda prompt, reasons, attempt: rule_based_rewrite(prompt, reasons)
    sch = mod("03_STRUCTURED_DATA.schemas")
    agents = mod("02_STORY_ENGINE.agents")
    provider = build_llm_provider(settings)
    agent = agents.PromptAgent(provider)

    def rewrite(prompt: str, reasons: list[str], attempt: int) -> str:
        current = sch.ImagePrompt.model_validate({**image_prompt_data, "prompt_text": prompt})
        try:
            return agent.rewrite_image_prompt(current, reasons, attempt).prompt_text
        except Exception as exc:  # LLM 改写失败就退回规则改写，不能让第二级本身把阶梯卡死
            log.warning("LLM 改写提示词失败，退回规则改写", extra={"error": str(exc)[:200]})
            return rule_based_rewrite(prompt, reasons)

    return rewrite


# ---- 主流程 ------------------------------------------------------------------------------


def produce_shot(
    payload: Dict[str, Any],
    *,
    settings: Settings | None = None,
    generator=None,
    scorer: Callable[..., Any] | None = None,
    rewriter: Callable[[str, list[str], int], str] | None = None,
) -> Dict[str, Any]:
    """纯函数版镜头生产（不依赖 Celery），shot_task 和测试都调它。

    payload: {"shot_id", "prompt", "negative_prompt", "reference_character_ids", "character_id",
      "episode_id", "seed", "output_dir", "run_id", "image_prompt": dict | None,
      "model": str | None, "license": str | None}
    """
    settings = settings or get_settings()
    errors_mod = mod("07_GENERATION.errors")
    ladder_mod = mod("07_GENERATION.retry_ladder")
    image_mod = mod("07_GENERATION.image.image_generator")
    asset_registry = mod("07_GENERATION.asset_registry")

    shot_id = payload["shot_id"]
    business_shot_id = payload.get("business_shot_id") or shot_id
    prompt = payload["prompt"]
    negative = payload.get("negative_prompt")
    character_ids = list(payload.get("reference_character_ids") or ([payload["character_id"]] if payload.get("character_id") else []))
    output_dir = Path(payload.get("output_dir") or settings.artifacts_root / "shots" / business_shot_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_log = str(output_dir.parent / "asset_records.jsonl")

    model = payload.get("model") or settings.image_backend
    if generator is None:
        generator = image_mod.DiffusersImageGenerator(model_id_or_path=model) if model and model != "dummy" else image_mod.DummyImageGenerator()
    scorer = scorer or (lambda **kw: score_asset(settings, **kw))
    rewriter = rewriter or make_prompt_rewriter(settings, payload.get("image_prompt"))

    with session_scope() as db:
        assets = resolve_character_assets(character_ids, db, project_id=payload.get("project_id"))
        repo.mark_shot_result(db, shot_id, status="generating")
    if assets["trigger_words"]:
        prompt = ", ".join(assets["trigger_words"]) + ", " + prompt

    ladder = ladder_mod.RetryLadder(resolution_ladder=[tuple(x) for x in settings.retry_resolution_ladder], max_attempts=max(2, settings.qc_max_attempts + 1))
    plan = ladder.first_attempt()
    seed = payload.get("seed") if payload.get("seed") is not None else random.randrange(2**31)
    reasons: list[str] = []
    attempts_log: list[dict] = []
    approved: dict[str, Any] | None = None

    while plan is not None:
        RETRY_LADDER.labels(plan.level.value).inc()
        if plan.rewrite_prompt:
            prompt = rewriter(prompt, reasons, plan.attempt)
        if plan.level == ladder_mod.RetryLevel.SAME_PARAMS:
            seed = random.randrange(2**31)  # "同参数"指 prompt/分辨率不变；换 seed 才有机会跳出坏采样
        failure = None
        try:
            image = generator.generate(
                prompt=prompt,
                negative_prompt=negative,
                seed=seed,
                width=plan.width,
                height=plan.height,
                lora_path=assets["lora_path"],
                reference_images=assets["reference_images"] or None,
            )
        except errors_mod.TransientProviderError as exc:
            failure, reasons = ladder_mod.FailureKind.TRANSIENT, [str(exc)]
        except errors_mod.GenerationRejectedError as exc:
            failure, reasons = ladder_mod.FailureKind.PROMPT_REJECTED, [str(exc)]
        except errors_mod.ResourceExhaustedError as exc:
            failure, reasons = ladder_mod.FailureKind.RESOURCE, [str(exc)]

        if failure is None:
            asset_id = asset_registry.new_asset_id("img")
            file_path = output_dir / f"{business_shot_id}_a{plan.attempt}.png"
            image.save(file_path)
            record = asset_registry.AssetRecord(
                asset_id=asset_id,
                file_path=str(file_path),
                character_id=character_ids[0] if character_ids else None,
                episode_id=payload.get("episode_id"),
                shot_id=shot_id,
                model=model or "dummy-placeholder-generator",
                prompt=prompt,
                seed=seed,
                license=payload.get("license"),
                created_at=asset_registry.now_iso(),
            )
            asset_registry.write_asset_record(record, asset_log)
            report = scorer(
                generated_image=str(file_path),
                reference_images=assets["reference_images"],
                shot_id=shot_id,
                asset_id=asset_id,
                character_id=character_ids[0] if character_ids else None,
            )
            passed = report.decision.value == "approved"
            QC_DECISIONS.labels(report.decision.value).inc()
            with session_scope() as db:
                repo.persist_asset(
                    db,
                    record,
                    kind="image",
                    qc_status="approved" if passed else "rejected",
                    attempt=plan.attempt,
                    retry_level=plan.level.value,
                    width=plan.width,
                    height=plan.height,
                )
                repo.persist_qc_report(db, report, attempt=plan.attempt)
            attempts_log.append({**plan.__dict__, "level": plan.level.value, "asset_id": asset_id, "qc": report.decision.value, "seed": seed})
            if passed:
                approved = {"asset_id": asset_id, "file_path": str(file_path), "width": plan.width, "height": plan.height, "seed": seed, "prompt": prompt}
                break
            failure, reasons = ladder_mod.FailureKind.QC_REJECTED, list(report.retry_reasons)
        else:
            attempts_log.append({**plan.__dict__, "level": plan.level.value, "error": reasons[0][:200] if reasons else ""})

        log.info("镜头生成未通过，进入重试阶梯", extra={"shot_id": shot_id, "attempt": plan.attempt, "failure": failure.value, "reasons": reasons[:2]})
        plan = ladder.next_attempt(failure, reason="; ".join(reasons)[:300])

    with session_scope() as db:
        if approved:
            repo.mark_shot_result(db, shot_id, status="approved", image_path=approved["file_path"])
        else:
            repo.mark_shot_result(db, shot_id, status="failed")

    return {
        "shot_id": shot_id,
        "business_shot_id": business_shot_id,
        "status": "approved" if approved else "failed",
        "asset": approved,
        "attempts": attempts_log,
        "ladder": ladder.summary(),
        "final_reasons": reasons if not approved else [],
    }


@celery_app.task(name="shot_task", bind=True)
def shot_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    return produce_shot(payload)
