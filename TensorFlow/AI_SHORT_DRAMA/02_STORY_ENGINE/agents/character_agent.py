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
from .memory import ConversationMemory


class CharacterAgent(BaseAgent):
    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "character_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3, memory: ConversationMemory | None = None):
        super().__init__(
            provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries, memory=memory
        )

    def generate_character(self, story_bible: sch.StoryBible, role_hint: str) -> sch.Character:
        user_prompt = (
            f"故事标题：{story_bible.title}\n主线冲突：{story_bible.main_conflict}\n"
            f"世界观：{story_bible.world.description}\n\n"
            f"请为这部短剧生成一个角色，角色定位提示：{role_hint}。"
            "需要包含完整的 appearance、personality_traits、backstory、motivation、relationships（至少 1 条）。"
        )
        return self.generate(user_prompt, sch.Character)
