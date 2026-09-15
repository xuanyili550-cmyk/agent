"""故事阶段任务：跑 02_STORY_ENGINE 的 LangGraph 流水线 -> 落库 -> 更新 PipelineRun。

一次任务处理一个 PipelineRun；参数全部来自 run.params：
    idea, num_characters, num_scenes, episode_numbers, prompt_mode ("llm"|"template"),
    planner ("llm"|"rule"), editorial (bool), target_duration_seconds
LLM provider / 记忆存储 / 用量记账都按 settings 走，任务本身不关心用的是 Claude 还是本地模型。

在流水线里的位置：这是第一环（run 状态 queued -> scripting -> scripted）。之后由编排层（13_INFRA.orchestration）
根据 require_human_review 决定是停在 awaiting_review 等人工批准，还是直接派发 shot_task。

一个 run 对应两份持久化状态，互为备份（见 ARCHITECTURE 2.16）：
1. **会话记忆** ``context_id = run:<run_id>``：所有 Agent 共用，存的是发给模型的 user/assistant 轮次，
   用来保证后面的 Agent 看得到前面 Agent 的产出，也用于事后审计"当初模型看到了什么"。
2. **LangGraph 检查点** ``thread_id = run:<run_id>``：存的是图的执行状态（跑完哪些节点、各节点产出的
   强类型对象）。worker 崩在第 5 个节点时，Celery 重试会从第 5 个节点续跑，前 4 个节点的 LLM 开销不再重复；
   已经跑完的 run 再被重试则直接跳过执行。两者用同一个 id，排查问题时能对上号。

流式：用 ``stream_pipeline`` 按节点执行，每完成一个节点就把进度写进 ``PipelineRun.result["progress"]``，
API 的 ``GET /pipelines/{run_id}/stream``（SSE）据此把进度实时推给前端，不用等整个阶段跑完。
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

__all__ = ["story_task", "run_story_stage", "pipeline_checkpointer"]

log = get_logger("queue.story")

# 故事图的六个节点，按执行顺序；SSE 进度条按它算百分比，前端不用自己写死一份
STORY_NODES = ["story_bible", "season_arc", "characters", "episodes", "storyboard", "prompts"]


def pipeline_checkpointer(settings, context_id: str):
    """按配置构造 LangGraph checkpointer。

    复用会话历史的存储后端（``build_conversation_store``）：生产是 Redis（多 worker 共享，
    换机器重试也能续跑），本地是 SQLite / JSON 文件。``context_id`` 只用来打日志。
    """
    workflows = mod("02_STORY_ENGINE.workflows")
    store = build_conversation_store(settings)
    log.debug("故事阶段检查点存储", extra={"context_id": context_id, "store": type(store).__name__})
    return workflows.build_checkpointer(store, lock_timeout=settings.llm_context_lock_timeout_seconds)


def _progress_writer(run_id: str):
    """返回 on_progress 回调：每完成一个节点就把进度写进 ``PipelineRun.result["progress"]``。

    为什么写数据库而不是推消息队列：进度要能被任意一个 API 副本读到（SSE 连在哪台机器都行），
    而且刷新页面后还能看到历史进度；写库是最简单又满足这两点的做法。
    写库失败只告警不中断——进度是观测数据，不该让它拖垮真正的生产任务。
    """

    def on_progress(event: dict) -> None:
        """把一条进度事件累加到 run.result.progress 里。"""
        if event["event"] == "done":
            return
        entry = {"node": event.get("node"), "event": event["event"], "keys": event.get("keys", []), "counts": event.get("counts", {}), "ts": event["ts"]}
        try:
            with session_scope() as db:
                run = db.get(PipelineRun, run_id)
                if run is None:
                    return
                progress = list((run.result or {}).get("progress") or [])
                progress.append(entry)
                done = [p["node"] for p in progress if p["event"] == "node" and p.get("node")]
                repo.update_run(
                    db,
                    run_id,
                    stage=f"story:{entry['node']}" if entry.get("node") else "story",
                    progress=progress,
                    progress_nodes=done,
                    progress_total=len(STORY_NODES),
                )
        except Exception as exc:
            log.warning("写流水线进度失败", extra={"run_id": run_id, "error": str(exc)[:200]})

    return on_progress


def run_story_stage(run_id: str) -> Dict[str, Any]:
    """故事阶段主体（不依赖 Celery，测试可直接调）：读 run.params -> 组装 Agent -> 跑图 -> 落库 -> 回写审核状态与结果。

    返回 {"run_id", "episode_ids"(数据库主键), "needing_review"(业务 id), "shots"}，供编排层决定下一步。
    """
    settings = get_settings()
    with session_scope() as db:
        run = db.get(PipelineRun, run_id)
        if run is None:
            raise LookupError(f"PipelineRun {run_id} 不存在")
        # 把需要的字段拷出来：出了 with 之后 ORM 行已 expire，不能再访问属性
        params = dict(run.params or {})
        project_id = run.project_id
        repo.update_run(db, run_id, status="scripting", stage="story")
    PIPELINE_RUNS.labels("scripting").inc()

    agents = mod("02_STORY_ENGINE.agents")
    planners = mod("02_STORY_ENGINE.planners")
    pipeline = mod("02_STORY_ENGINE.workflows.pipeline")

    provider = build_llm_provider(settings)
    # 一个 run 一个会话：所有 Agent 共享历史，重跑同一个 run 也能接着之前的上下文
    context_id = f"run:{run_id}"
    memory = agents.build_memory(provider, context_id, build_conversation_store(settings))
    memory.lock_timeout = settings.llm_context_lock_timeout_seconds
    sink = make_usage_sink(settings, context_id=context_id, run_id=run_id)

    num_episodes_planned = int(params.get("num_episodes_planned", 12))
    use_llm_planner = params.get("planner", "llm") == "llm"
    editorial = bool(params.get("editorial", True))

    def agent(cls, **kw):
        """构造一个 Agent 并挂上共享 memory 和用量回调；所有 Agent 都这样建，避免漏挂 usage_sink 导致用量少记。"""
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
        # 季度规划可选 LLM（ChapterPlannerAgent）或规则（SeasonArcPlanner）；规则版不花钱、确定性强，测试和模板模式用
        season_planner=agent(agents.ChapterPlannerAgent, num_episodes=num_episodes_planned)
        if use_llm_planner
        else planners.SeasonArcPlanner(num_episodes=num_episodes_planned),
        episode_planner=planners.EpisodePlanner(),
        # editorial=False 时不配质检官/总编审，图里的编审节点会直接跳过（status=skipped）
        qc_officer=agent(agents.QCOfficerAgent) if editorial else None,
        chief_editor=agent(agents.ChiefEditorAgent) if editorial else None,
        max_revision_rounds=int(params.get("max_revision_rounds", 2)),
    )
    # 检查点和会话历史用同一个存储后端（Redis / SQLite / 本地文件，由 LLM_CONTEXT_* 决定），
    # 用同一个 id（run:<run_id>）；关掉 STORY_CHECKPOINT_ENABLED 就退回无持久化的一次性执行。
    checkpointer = pipeline_checkpointer(settings, context_id) if settings.story_checkpoint_enabled else None
    graph = pipeline.build_graph(
        bundle,
        num_characters=int(params.get("num_characters", 3)),
        num_scenes=int(params.get("num_scenes", 2)),
        episode_numbers=[int(n) for n in params.get("episode_numbers", [1])],
        prompt_mode=params.get("prompt_mode", "llm"),
        checkpointer=checkpointer,
    )
    initial_state = {"idea": params["idea"], "target_duration_seconds": int(params.get("target_duration_seconds", 300))}
    if checkpointer is None:
        state = graph.invoke(initial_state)
    else:
        before = pipeline.graph_progress(graph, context_id)
        if before["completed"]:
            log.info("故事阶段从检查点续跑", extra={"run_id": run_id, "completed": before["completed"], "pending": before["pending"]})
        state = pipeline.run_pipeline(graph, initial_state, context_id, on_progress=_progress_writer(run_id))

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
                # 被 AI 打回的集也标 pending_review 而不是 rejected：最终否决权在人，AI 只提供意见（notes）
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
            # 只数六个生产 Agent 的调用次数；编审 Agent 的调用在 llm_usage 表里有，这里给个快速概览
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
    """Celery 入口：跑 run_story_stage；任何异常都先把 run 标成 failed（带错误摘要）再抛出，
    API 查 run 状态时才不会看到一个永远卡在 scripting 的运行。"""
    try:
        return run_story_stage(run_id)
    except Exception as exc:
        with session_scope() as db:
            repo.update_run(db, run_id, status="failed", stage="story", error=f"{type(exc).__name__}: {str(exc)[:500]}")
        PIPELINE_RUNS.labels("failed").inc()
        raise
