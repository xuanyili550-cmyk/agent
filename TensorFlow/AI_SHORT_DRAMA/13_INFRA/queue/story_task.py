"""故事阶段任务：跑 02_STORY_ENGINE 的 LangGraph 流水线 -> 落库 -> 更新 PipelineRun。

一次任务处理一个 PipelineRun；参数全部来自 run.params：
    idea, num_characters, num_scenes, episode_numbers, prompt_mode ("llm"|"template"),
    planner ("llm"|"rule"), editorial (bool), target_duration_seconds
LLM provider / 记忆存储 / 用量记账都按 settings 走，任务本身不关心用的是 Claude 还是本地模型。
"""

from __future__ import annotations

from typing import Any, Dict

from ..config import get_settings
from ..database import repository as repo
from ..database.models import PipelineRun
from ..database.session import session_scope
from ..observability import PIPELINE_RUNS, get_logger
from ..workers.celery_app import celery_app
from ._common import build_conversation_store, build_llm_provider, make_usage_sink, mod

__all__ = ["story_task", "run_story_stage"]

log = get_logger("queue.story")


def run_story_stage(run_id: str) -> Dict[str, Any]:
    settings = get_settings()
    with session_scope() as db:
        run = db.get(PipelineRun, run_id)
        if run is None:
            raise LookupError(f"PipelineRun {run_id} 不存在")
        params = dict(run.params or {})
        project_id = run.project_id
        repo.update_run(db, run_id, status="scripting", stage="story")
    PIPELINE_RUNS.labels("scripting").inc()

    agents = mod("02_STORY_ENGINE.agents")
    planners = mod("02_STORY_ENGINE.planners")
    pipeline = mod("02_STORY_ENGINE.workflows.pipeline")

    provider = build_llm_provider(settings)
    context_id = f"run:{run_id}"
    memory = agents.build_memory(provider, context_id, build_conversation_store(settings))
    memory.lock_timeout = settings.llm_context_lock_timeout_seconds
    sink = make_usage_sink(settings, context_id=context_id, run_id=run_id)

    num_episodes_planned = int(params.get("num_episodes_planned", 12))
    use_llm_planner = params.get("planner", "llm") == "llm"
    editorial = bool(params.get("editorial", True))

    def agent(cls, **kw):
        a = cls(provider, memory=memory, **kw)
        a.usage_sink = sink
        return a

    bundle = pipeline.AgentBundle(
        story_agent=agent(agents.StoryAgent),
        character_agent=agent(agents.CharacterAgent),
        episode_agent=agent(agents.EpisodeAgent),
        screenplay_agent=agent(agents.ScreenplayAgent),
        storyboard_agent=agent(agents.StoryboardAgent),
        prompt_agent=agent(agents.PromptAgent),
        season_planner=agent(agents.ChapterPlannerAgent, num_episodes=num_episodes_planned)
        if use_llm_planner
        else planners.SeasonArcPlanner(num_episodes=num_episodes_planned),
        episode_planner=planners.EpisodePlanner(),
        qc_officer=agent(agents.QCOfficerAgent) if editorial else None,
        chief_editor=agent(agents.ChiefEditorAgent) if editorial else None,
        max_revision_rounds=int(params.get("max_revision_rounds", 2)),
    )
    graph = pipeline.build_graph(
        bundle,
        num_characters=int(params.get("num_characters", 3)),
        num_scenes=int(params.get("num_scenes", 2)),
        episode_numbers=[int(n) for n in params.get("episode_numbers", [1])],
        prompt_mode=params.get("prompt_mode", "llm"),
    )
    state = graph.invoke({"idea": params["idea"], "target_duration_seconds": int(params.get("target_duration_seconds", 300))})

    with session_scope() as db:
        written = repo.persist_drama_state(db, project_id, state)
        needing_review = list(state.get("episodes_needing_review", []))
        editorial_records = state.get("editorial_by_episode", {})
        # AI 编审判定回写到 episodes.review_status：approved 的集在 ai_editor_can_approve=true 时不再等人
        sid = lambda business_id: repo.scoped_id(project_id, business_id)  # noqa: E731
        for ep in state.get("episodes", []):
            rec = editorial_records.get(ep.id, {})
            row = db.get(mod("13_INFRA.database.models").Episode, sid(ep.id))
            if row is None:
                continue
            if rec.get("status") == "approved" and settings.ai_editor_can_approve:
                row.review_status = "approved"
            elif rec.get("status") in ("rejected", "needs_human_review"):
                row.review_status = "pending_review"
                row.review_notes = (rec.get("editorial") or {}).get("notes")
            else:
                row.review_status = "pending_review"
        first_episode = sid(state["episodes"][0].id) if state.get("episodes") else None
        run = repo.update_run(
            db,
            run_id,
            status="scripted",
            stage="story_done",
            written=written,
            episode_ids=[sid(ep.id) for ep in state.get("episodes", [])],
            episodes_needing_review=[sid(e) for e in needing_review],
            editorial={sid(k): {"status": v["status"], "rounds": v["revision_rounds"]} for k, v in editorial_records.items()},
            shots_total=len(state.get("shots", [])),
            llm_calls=sum(
                len(a.usage_log)
                for a in [
                    bundle.story_agent,
                    bundle.character_agent,
                    bundle.episode_agent,
                    bundle.screenplay_agent,
                    bundle.storyboard_agent,
                    bundle.prompt_agent,
                ]
                if a
            ),
        )
        run.episode_id = first_episode
    log.info("故事阶段完成", extra={"run_id": run_id, "episodes": len(state.get("episodes", [])), "shots": len(state.get("shots", []))})
    return {
        "run_id": run_id,
        "episode_ids": [repo.scoped_id(project_id, ep.id) for ep in state.get("episodes", [])],
        "needing_review": needing_review,
        "shots": len(state.get("shots", [])),
    }


@celery_app.task(name="story_task", bind=True)
def story_task(self, run_id: str) -> Dict[str, Any]:
    try:
        return run_story_stage(run_id)
    except Exception as exc:
        with session_scope() as db:
            repo.update_run(db, run_id, status="failed", stage="story", error=f"{type(exc).__name__}: {str(exc)[:500]}")
        PIPELINE_RUNS.labels("failed").inc()
        raise
