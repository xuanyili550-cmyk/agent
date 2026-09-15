"""故事 Agent：负责根据一句话创意生成完整的 Story Bible（故事圣经）。

Story Bible 是整条短剧生产流水线的起点，后续的季度大纲（SeasonArcPlanner）、
分集规划（EpisodePlanner）、分场/分镜生成（StoryboardAgent）都依赖它提供的
世界观、题材、基调、目标受众、核心冲突等信息。之所以单独拆出一个 Agent 类，
是因为生成 Story Bible 需要一段专属的系统提示词（system prompt）和独立的
结构化输出 schema（sch.StoryBible），与其他 Agent 的职责不同，拆分后互不干扰、
便于分别调优提示词。
"""

from __future__ import annotations

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        # 向上遍历父目录找到项目根，把结构化数据模块目录和本模块目录插入
        # sys.path，这样无论从哪个工作目录启动脚本都能正常 import schemas。
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

import schemas as sch

from .base import PROMPTS_DIR, BaseAgent, LLMProvider
from .memory import ConversationMemory


class StoryAgent(BaseAgent):
    """负责把一句话故事创意扩展为结构化 Story Bible 的 Agent。"""

    # 系统提示词文件路径：指向 prompts 目录下专门为 StoryAgent 编写的提示词模板。
    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "story_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3, memory: ConversationMemory | None = None):
        """初始化 StoryAgent，读取专属系统提示词并交给父类完成 LLM 调用相关的通用初始化。"""
        super().__init__(provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries, memory=memory)

    def generate_story_bible(self, idea: str) -> sch.StoryBible:
        """根据故事创意生成完整的 Story Bible。

        在用户提示词里明确列出必须包含的字段（world/genre/themes 等），
        目的是约束 LLM 输出符合 sch.StoryBible 结构化 schema，减少解析失败率。
        """
        user_prompt = (
            f"故事创意：{idea}\n\n"
            "请基于以上创意生成一部竖屏短剧的完整 Story Bible，包含 world（世界观设定，含至少 2 个 location）、"
            "genre、themes、tone、target_audience、main_conflict、unique_selling_point、episode_count_planned、character_ids。"
        )
        return self.generate(user_prompt, sch.StoryBible)
