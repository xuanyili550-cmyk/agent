"""章节规划师：用 LLM 把 Story Bible 规划成 SeasonArc（幕 + 每集归属）。

和 planners/season_arc_planner.py 的规则版是互补关系：规则版零成本、结果确定，适合测试和
兜底；LLM 版能按具体剧情安排反转节奏。两者输出同一个 schemas.SeasonArc，下游不感知差异。
LLM 版的输出会经过 ``_reconcile()`` 硬校验：集 id 必须与给定列表完全一致，
否则用规则版兜底——章节规划错一个 id，后面所有集都会串。
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
from planners.season_arc_planner import SeasonArcPlanner

from .base import PROMPTS_DIR, BaseAgent, LLMProvider
from .memory import ConversationMemory


class ChapterPlannerAgent(BaseAgent):
    """把 Story Bible 规划成季线（幕结构 + 每幕包含哪些集）的 LLM Agent。"""

    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "chapter_planner_agent_system.txt"

    def __init__(
        self,
        provider: LLMProvider,
        max_retries: int = 3,
        memory: ConversationMemory | None = None,
        num_episodes: int = 12,
        num_acts: int = 3,
    ):
        """记下集数/幕数配置，并同时准备好规则版 ``SeasonArcPlanner`` 作为兜底。"""
        super().__init__(provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries, memory=memory)
        self.num_episodes = num_episodes
        self.num_acts = num_acts
        self._fallback = SeasonArcPlanner(num_episodes=num_episodes, num_acts=num_acts)

    def plan(self, story_bible: sch.StoryBible, season_number: int = 1) -> sch.SeasonArc:
        """让 LLM 按 Story Bible 规划一季，再交给 ``_reconcile`` 做集 id 完整性校验。"""
        episode_ids = [f"ep_{i:03d}" for i in range(1, self.num_episodes + 1)]
        user_prompt = (
            f"Story Bible：\n标题：{story_bible.title}\nlogline：{story_bible.logline}\n"
            f"核心冲突：{story_bible.main_conflict}\n题材：{'、'.join(g.value for g in story_bible.genre)}\n"
            f"主题：{'、'.join(story_bible.themes)}\n卖点：{story_bible.unique_selling_point}\n\n"
            f"请规划第 {season_number} 季，共 {self.num_episodes} 集、{self.num_acts} 幕。"
            f"id 使用 'season_{season_number:03d}'，season_number={season_number}，"
            f"episode_ids 必须恰好是（按顺序）：{', '.join(episode_ids)}；"
            "beats 里每一幕的 episode_ids 从这个列表按顺序切分，不重不漏。"
        )
        arc = self.generate(user_prompt, sch.SeasonArc)
        return self._reconcile(arc, story_bible, season_number, episode_ids)

    def _reconcile(self, arc: sch.SeasonArc, bible: sch.StoryBible, season_number: int, episode_ids: list[str]) -> sch.SeasonArc:
        """校验 LLM 分配的集 id 是否与预期列表完全一致，不一致就换规则版兜底。"""
        assigned = [eid for beat in arc.beats for eid in beat.episode_ids]
        if assigned == episode_ids and arc.episode_ids == episode_ids and len(arc.beats) == self.num_acts:
            return arc
        # LLM 把集 id 分错了：保留它写的幕描述/中心问题，集分配换成规则版的确定结果
        fallback = self._fallback.plan(bible, season_number)
        beats = []
        for i, fb in enumerate(fallback.beats):
            llm_beat = arc.beats[i] if i < len(arc.beats) else None
            beats.append(
                sch.StoryBeat(
                    act=llm_beat.act if llm_beat else fb.act,
                    description=llm_beat.description if llm_beat else fb.description,
                    episode_ids=fb.episode_ids,
                )
            )
        return fallback.model_copy(
            update={
                "title": arc.title or fallback.title,
                "synopsis": arc.synopsis or fallback.synopsis,
                "central_question": arc.central_question or fallback.central_question,
                "resolution": arc.resolution or fallback.resolution,
                "beats": beats,
            }
        )
