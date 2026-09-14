"""总编审：拿着质检官报告对一集做最终决定（approve / revise / reject）。

reject 或多轮 revise 仍不通过时，流水线停在 awaiting_review 等人工；approve 才进生产。
质检官报告里有 blocker 时，代码层面强制不允许 approve——不能指望 LLM 每次都遵守 prompt。
"""

from __future__ import annotations

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

import schemas as sch

from .base import PROMPTS_DIR, BaseAgent, LLMProvider
from .memory import ConversationMemory


class ChiefEditorAgent(BaseAgent):
    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "chief_editor_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3, memory: ConversationMemory | None = None):
        super().__init__(provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries, memory=memory)

    def decide(
        self,
        story_bible: sch.StoryBible,
        season_arc: sch.SeasonArc,
        episode: sch.Episode,
        script: sch.Script,
        qc_verdict: sch.QCVerdict,
        revision_round: int = 0,
    ) -> sch.EditorialDecision:
        beat = next((b for b in season_arc.beats if episode.id in b.episode_ids), None)
        issues = "\n".join(f"- [{i.severity.value}] {i.code} @ {i.location}: {i.description}（建议：{i.suggestion}）" for i in qc_verdict.issues) or "（无）"
        user_prompt = (
            f"Story Bible：《{story_bible.title}》{story_bible.logline}\n目标观众：{story_bible.target_audience}\n"
            f"季线：{season_arc.central_question}\n本集所属幕：{beat.act if beat else '未知'} —— {beat.description if beat else ''}\n\n"
            f"待审集：{episode.id}《{episode.title}》\nhook：{episode.hook}\ncliffhanger：{episode.cliffhanger}\n"
            f"梗概：{episode.synopsis}\n情绪弧线：{episode.emotional_arc}\n\n剧本正文：\n{script.content[:3000]}\n\n"
            f"质检官报告：passed={qc_verdict.passed} score={qc_verdict.score}\n{qc_verdict.summary}\n问题清单：\n{issues}\n\n"
            f"这是第 {revision_round} 轮修订后的版本。请给出最终决定，id 使用 'ed_{episode.id}'，episode_id 为 '{episode.id}'。"
        )
        decision = self.generate(user_prompt, sch.EditorialDecision)
        has_blocker = any(i.severity == sch.IssueSeverity.BLOCKER for i in qc_verdict.issues)
        if has_blocker and decision.decision == sch.EditorialDecisionType.APPROVE:
            decision = decision.model_copy(
                update={
                    "decision": sch.EditorialDecisionType.REVISE,
                    "required_changes": decision.required_changes + [i.description for i in qc_verdict.issues if i.severity == sch.IssueSeverity.BLOCKER],
                    "notes": decision.notes + "\n（系统：质检官报告含 blocker，自动改为 revise）",
                }
            )
        return decision.model_copy(update={"id": f"ed_{episode.id}", "episode_id": episode.id})
