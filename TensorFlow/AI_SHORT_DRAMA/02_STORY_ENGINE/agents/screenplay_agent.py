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


class ScreenplayAgent(BaseAgent):
    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "screenplay_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3, memory: ConversationMemory | None = None):
        super().__init__(
            provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries, memory=memory
        )

    def generate_script(self, episode: sch.Episode, characters: list[sch.Character]) -> sch.Script:
        cast = "、".join(c.name for c in characters)
        user_prompt = (
            f"集标题：{episode.title}\n本集梗概：{episode.synopsis}\n"
            f"开场钩子：{episode.hook}\n结尾悬念：{episode.cliffhanger}\n"
            f"出场角色：{cast}\n\n"
            f"请为 episode_id='{episode.id}' 生成完整剧本文本（content 字段为带场景标题、动作描述、台词的完整剧本正文，"
            "语言为中文），id 使用 'script_' 加集号。"
        )
        return self.generate(user_prompt, sch.Script)
