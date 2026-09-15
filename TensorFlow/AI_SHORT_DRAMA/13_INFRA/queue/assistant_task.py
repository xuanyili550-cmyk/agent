"""制片助理任务：在一次流水线运行的产出上跑 DramaAssistantAgent（带工具循环的 Agent）。

    POST /pipelines/{run_id}/assistant  {"question": "..."}  ->  assistant_task(payload)

任务体做三件事（对应 ARCHITECTURE 2.6 "队列任务只做调度 + 落盘 + 记账"）：
1. 从数据库把这次运行的故事设定 / 角色 / 集 / 剧本 / 场次 / 镜头读回 03 schema 对象，装成 DramaContext；
2. 构造 Agent 并接上 13_INFRA 的四个回调（token 记账、工具调用指标、run 指标、确认门白名单）；
3. 跑 ``agent.ask(question)``，把答案、每一步工具调用记录、用量返回给 API。

唯一的"写"操作是工具 ``flag_episode_for_human_review``：把集的 review_status 打回 pending_review
并写 review_notes。它标了 requires_confirmation，只有 ``AGENT_TOOL_ALLOWLIST`` 里放行了才会执行，
否则 Agent 会收到"被拒绝"并如实告诉用户——这就是"工具级确认门"在后台任务里的形态。
"""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import select

from ..config import get_settings
from ..database import repository as repo
from ..database.models import Character, Episode, PipelineRun, StoryBible
from ..database.session import session_scope
from ..observability import get_logger
from ..workers.celery_app import celery_app
from ._common import build_conversation_store, build_llm_provider, estimate_cost_usd, instrument_tool_agent, mod

__all__ = ["assistant_task", "build_context_from_db"]

log = get_logger("queue.assistant")


def build_context_from_db(run_id: str):
    """把一次运行落库的产出读成 DramaContext（02_STORY_ENGINE.agents.DramaContext）。

    集 / 场次 / 镜头都以业务 id（ep_001 / scene_001_01）为键，和 02 里 LangGraph 直接产出的对象一致，
    这样 build_drama_toolbox 的工具代码在 demo 和生产里是同一份。
    """
    sch = mod("03_STRUCTURED_DATA.schemas")
    assistant_mod = mod("02_STORY_ENGINE.agents.drama_assistant")

    with session_scope() as db:
        run = db.get(PipelineRun, run_id)
        if run is None:
            raise LookupError(f"PipelineRun {run_id} 不存在")
        project_id = run.project_id
        episode_db_ids = list((run.result or {}).get("episode_ids") or [])

        bible_row = db.execute(select(StoryBible).where(StoryBible.project_id == project_id)).scalars().first()
        story_bible = sch.StoryBible.model_validate(bible_row.data) if bible_row and bible_row.data else None
        characters = [
            sch.Character.model_validate(row.data)
            for row in db.execute(select(Character).where(Character.project_id == project_id)).scalars().all()
            if row.data
        ]

        episodes, scripts, scenes, shots, editorial = [], {}, {}, {}, {}
        for ep_db_id in episode_db_ids:
            row = db.get(Episode, ep_db_id)
            if row is None or not row.data:
                continue
            episode = sch.Episode.model_validate(row.data)
            episodes.append(episode)
            if row.script:
                scripts[episode.id] = sch.Script.model_validate(row.script)
            if row.editorial:
                editorial[episode.id] = dict(row.editorial)
            scenes[episode.id] = repo.load_scenes_for_episode(db, ep_db_id)
            shots[episode.id] = [shot for _db_id, shot in repo.load_shots_for_episode(db, ep_db_id)]

    def flag_handler(episode_id: str, reason: str) -> None:
        """工具 flag_episode_for_human_review 的落库实现：打回待审并记录原因。"""
        with session_scope() as db:
            row = db.get(Episode, repo.scoped_id(project_id, episode_id))
            if row is None:
                raise LookupError(f"数据库里没有集 {episode_id}")
            row.review_status = "pending_review"
            row.review_notes = f"[制片助理] {reason}"
        log.info("制片助理标记人工复核", extra={"run_id": run_id, "episode_id": episode_id, "reason": reason[:200]})

    return assistant_mod.DramaContext(
        story_bible=story_bible,
        characters=characters,
        episodes=episodes,
        scripts=scripts,
        scenes=scenes,
        shots=shots,
        editorial=editorial,
        flag_handler=flag_handler,
    )


@celery_app.task(name="assistant_task", bind=True)
def assistant_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """payload: {"run_id": str, "question": str, "context_id": str | None}。

    ``context_id`` 默认 ``assistant:<run_id>``：同一次运行的多次提问共享会话记忆，
    第二个问题能接着第一个问题的上下文。瞬时 LLM 错误由 BaseTask 指数退避重试。
    """
    settings = get_settings()
    run_id = payload["run_id"]
    question = payload["question"]
    context_id = payload.get("context_id") or f"assistant:{run_id}"

    agents = mod("02_STORY_ENGINE.agents")
    provider = build_llm_provider(settings)
    ctx = build_context_from_db(run_id)
    memory = agents.build_memory(provider, context_id, build_conversation_store(settings))
    memory.lock_timeout = settings.llm_context_lock_timeout_seconds

    agent = agents.DramaAssistantAgent(provider, ctx, memory=memory)
    instrument_tool_agent(agent, settings, context_id=context_id, run_id=run_id)
    result = agent.ask(question)

    answer = result.answer if isinstance(result.answer, str) else str(result.answer)
    log.info("制片助理回答完成", extra={"run_id": run_id, "tool_rounds": result.tool_rounds, "stopped_reason": result.stopped_reason})
    return {
        "run_id": run_id,
        "context_id": context_id,
        "answer": answer,
        "steps": [
            {"tool": s.tool, "args": s.args, "status": s.status, "error": s.error, "duration_seconds": round(s.duration_seconds, 4)} for s in result.steps
        ],
        "tool_rounds": result.tool_rounds,
        "stopped_reason": result.stopped_reason,
        "flagged": list(ctx.flagged),
        "usage": {
            "llm_calls": len(result.usage),
            "input_tokens": sum(u.input_tokens for u in result.usage),
            "output_tokens": sum(u.output_tokens for u in result.usage),
            "cost_usd": round(sum(estimate_cost_usd(settings, u.model, u.input_tokens, u.output_tokens) for u in result.usage), 6),
        },
    }
