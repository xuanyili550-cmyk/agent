from __future__ import annotations

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

import schemas as sch

from .base import BaseAgent, LLMProvider, PROMPTS_DIR


class StoryAgent(BaseAgent):
    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "story_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3):
        super().__init__(provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries)

    def generate_story_bible(self, idea: str) -> sch.StoryBible:
        user_prompt = (
            f"故事创意：{idea}\n\n"
            "请基于以上创意生成一部竖屏短剧的完整 Story Bible，包含 world（世界观设定，含至少 2 个 location）、"
            "genre、themes、tone、target_audience、main_conflict、unique_selling_point、episode_count_planned、character_ids。"
        )
        return self.generate(user_prompt, sch.StoryBible)
