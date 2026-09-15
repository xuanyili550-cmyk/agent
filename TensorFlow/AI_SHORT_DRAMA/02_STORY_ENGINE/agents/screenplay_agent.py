"""编剧 Agent：把一集的梗概/钩子/悬念/角色表，生成为一份完整的中文剧本文本。

在流水线里处于故事大纲之后、拆分镜头之前：上游是 Episode/角色表这些结构化设定，下游是
分镜拆解和质检。之所以要经过一版完整剧本文本而不是直接让 LLM 生成分镜结构，是因为剧本正文
是人（编审）最容易读懂和提修改意见的中间产物，也是后续分镜拆解时"照着剧本走"的依据；
支持带着上一版剧本 + 修改意见重新生成，对应总编审/质检官打回重写的场景。
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


class ScreenplayAgent(BaseAgent):
    """根据集设定（及可选的上一版剧本与修改意见）生成完整剧本文本的 Agent。"""

    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "screenplay_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3, memory: ConversationMemory | None = None):
        """加载系统提示词文件并初始化基类；参数含义见 ``BaseAgent.__init__``。"""
        super().__init__(provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries, memory=memory)

    def generate_script(
        self,
        episode: sch.Episode,
        characters: list[sch.Character],
        previous_script: sch.Script | None = None,
        revision_notes: list[str] | None = None,
    ) -> sch.Script:
        """生成（或按修改意见重写）一集的完整剧本。

        无 ``revision_notes`` 时是从集设定直接生成第一版（version=1）；有修改意见时视为
        编审/质检官打回重写，version 在上一版基础上 +1，并把上一版正文和全部修改意见一起
        交给 LLM，要求"在原稿基础上改"而不是重新编一个故事，避免推倒重来导致已通过的部分
        又出新问题。
        """
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
        # version/episode_id 由代码强制写回，不依赖 LLM 输出是否准确（防止模型把版本号算错）
        return script.model_copy(update={"version": version, "episode_id": episode.id})
