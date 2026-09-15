"""角色设计师：用 LLM 按 Story Bible 和角色定位提示生成一个完整角色（外貌/性格/背景/动机/关系）。

流水线里角色生成在 Story Bible 之后、分集剧本之前：后续 episode/script 生成都要引用
这里产出的角色 id 和设定，保证同一部短剧里角色形象前后一致。
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


class CharacterAgent(BaseAgent):
    """根据 Story Bible 和角色定位提示，生成单个角色的结构化设定。"""

    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "character_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3, memory: ConversationMemory | None = None):
        """直接复用 BaseAgent 的初始化，加载本 agent 专属的系统提示词文件。"""
        super().__init__(provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries, memory=memory)

    def generate_character(self, story_bible: sch.StoryBible, role_hint: str) -> sch.Character:
        """按给定的角色定位提示（如"男主""反派"）生成一个角色，字段要求写在 prompt 里强制补全。"""
        user_prompt = (
            f"故事标题：{story_bible.title}\n主线冲突：{story_bible.main_conflict}\n"
            f"世界观：{story_bible.world.description}\n\n"
            f"请为这部短剧生成一个角色，角色定位提示：{role_hint}。"
            "需要包含完整的 appearance、personality_traits、backstory、motivation、relationships（至少 1 条）。"
        )
        return self.generate(user_prompt, sch.Character)
