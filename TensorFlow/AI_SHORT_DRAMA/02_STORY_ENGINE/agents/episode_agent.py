from __future__ import annotations

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

import schemas as sch

from .base import PROMPTS_DIR, AgentGenerationError, BaseAgent, LLMProvider
from .memory import ConversationMemory


def previous_episodes_recap(previous: list[sch.Episode], limit: int = 3) -> str:
    """把前几集压成"前情提要"文本：多集连续生成时，本集必须接着上一集的 cliffhanger 写。"""
    if not previous:
        return ""
    recent = previous[-limit:]
    lines = [f"第 {ep.episode_number} 集《{ep.title}》：{ep.synopsis} 结尾悬念：{ep.cliffhanger}" for ep in recent]
    return "前情提要（本集开场必须承接最后一条悬念）：\n" + "\n".join(lines) + "\n"


class EpisodeAgent(BaseAgent):
    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "episode_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3, memory: ConversationMemory | None = None):
        super().__init__(provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries, memory=memory)

    def generate_episode(
        self,
        season_arc: sch.SeasonArc,
        episode_number: int,
        beat: sch.StoryBeat,
        constraints: dict,
        previous_episodes: list[sch.Episode] | None = None,
        revision_notes: list[str] | None = None,
    ) -> sch.Episode:
        recap = previous_episodes_recap(previous_episodes or [])
        revision = ""
        if revision_notes:
            revision = "总编审要求修改（必须全部落实）：\n" + "\n".join(f"- {n}" for n in revision_notes) + "\n"
        user_prompt = (
            f"季线标题：{season_arc.title}\n季线核心问题：{season_arc.central_question}\n"
            f"当前所属幕：{beat.act} —— {beat.description}\n"
            f"{recap}{revision}"
            f"这是第 {episode_number} 集，season_id 为 '{season_arc.id}'，id 请使用 'ep_{episode_number:03d}'。\n"
            f"时长约束：duration_seconds 必须在 {constraints['target_duration_seconds']} 秒左右。\n"
            # 短剧核心叙事约束：开场必须在极短时间内建立强钩子，结尾必须留强悬念驱动下一集点击
            f"强制约束：hook 字段必须描述开场 {constraints['hook_window_seconds']} 秒内建立的强钩子（冲突/反转/悬疑画面），"
            "cliffhanger 字段必须描述结尾的强悬念，两者都不能为空。"
        )
        episode = self.generate(user_prompt, sch.Episode)
        if not episode.hook.strip() or not episode.cliffhanger.strip():
            raise AgentGenerationError("生成的 Episode 缺少 hook 或 cliffhanger，违反短剧强钩子/强悬念约束")
        return episode
