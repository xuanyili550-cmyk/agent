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


class ScreenplayAgent(BaseAgent):
    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "screenplay_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3, memory: ConversationMemory | None = None):
        super().__init__(provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries, memory=memory)

    def generate_script(
        self,
        episode: sch.Episode,
        characters: list[sch.Character],
        previous_script: sch.Script | None = None,
        revision_notes: list[str] | None = None,
    ) -> sch.Script:
        cast = "、".join(f"{c.name}（{c.id}，{c.role.value}）" for c in characters)
        revision = ""
        version = 1
        if revision_notes:
            # 编审打回：把上一版剧本和修改意见一起给模型，要求在原稿基础上改而不是重写一个新故事
            version = (previous_script.version + 1) if previous_script else 2
            prior = f"上一版剧本（version {previous_script.version}）：\n{previous_script.content[:3000]}\n\n" if previous_script else ""
            revision = prior + "总编审/质检官要求修改（必须全部落实，其余保持不变）：\n" + "\n".join(f"- {n}" for n in revision_notes) + "\n\n"
        user_prompt = (
            f"集标题：{episode.title}\n本集梗概：{episode.synopsis}\n"
            f"开场钩子：{episode.hook}\n结尾悬念：{episode.cliffhanger}\n"
            f"出场角色：{cast}\n\n{revision}"
            f"请为 episode_id='{episode.id}' 生成完整剧本文本（content 字段为带场景标题、动作描述、台词的完整剧本正文，"
            f"语言为中文），id 使用 'script_' 加集号，version 填 {version}。"
        )
        script = self.generate(user_prompt, sch.Script)
        return script.model_copy(update={"version": version, "episode_id": episode.id})
