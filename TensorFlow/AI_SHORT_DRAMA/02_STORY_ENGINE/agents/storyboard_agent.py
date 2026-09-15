"""分镜 Agent：负责把剧本（Script）逐步细化为场次（Scene）、镜头（Shot）与对白（DialogueLine）。

这是流水线中承上启下的一环：上游拿到 StoryAgent/SeasonArcPlanner 产出的剧情结构后，
StoryboardAgent 把「文字剧本」转成可以直接喂给图像/视频生成模块的「结构化分镜数据」。
之所以把生成分场、生成镜头、生成对白拆成三个独立方法，是因为它们依赖的上下文粒度
不同（场次描述 -> 镜头列表 -> 对白），分步生成可以让每次 LLM 调用的输出更聚焦、
更容易约束到目标 schema，也便于单独重试某一步而不用重跑整个分镜流程。
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

from pydantic import BaseModel, ConfigDict

import schemas as sch

from .base import PROMPTS_DIR, BaseAgent, LLMProvider
from .memory import ConversationMemory


class DialogueBatch(BaseModel):
    """一批对白的容器模型，仅用于约束 LLM 输出为 {"dialogue": [...]} 这种结构。"""

    # 禁止额外字段，避免 LLM 输出多余键导致解析歧义。
    model_config = ConfigDict(extra="forbid")

    dialogue: list[sch.DialogueLine]


class StoryboardAgent(BaseAgent):
    """负责生成分场、分镜镜头、对白的 Agent。"""

    # 系统提示词文件路径：指向 prompts 目录下专门为 StoryboardAgent 编写的提示词模板。
    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "storyboard_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3, memory: ConversationMemory | None = None):
        """初始化 StoryboardAgent，读取专属系统提示词并交给父类完成 LLM 调用相关的通用初始化。"""
        super().__init__(provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries, memory=memory)

    def generate_scene(self, episode: sch.Episode, scene_number: int, script: sch.Script) -> sch.Scene:
        """根据剧本内容为指定集数生成一场戏的场次信息（不含镜头和对白，留待后续步骤填充）。"""
        user_prompt = (
            f"剧本内容：{script.content[:2000]}\n\n"
            f"请为 episode_id='{episode.id}' 生成第 {scene_number} 场（scene_number={scene_number}）的分场信息，"
            f"id 使用 'scene_{episode.episode_number:03d}_{scene_number:02d}'。shot_ids 和 dialogue 先留空列表，"
            "后续会单独生成分镜与对白。"
        )
        return self.generate(user_prompt, sch.Scene)

    def generate_shots(self, scene: sch.Scene, episode_number: int) -> list[sch.Shot]:
        """根据场次描述生成该场的分镜镜头列表（2-4 个），每个镜头附带机位、动作、情绪等信息。"""
        user_prompt = (
            f"场次描述：{scene.description}\n出场角色：{'、'.join(scene.characters_present)}\n"
            f"请为 scene_id='{scene.id}'（episode={episode_number}, scene={scene.scene_number}）生成 2-4 个分镜镜头，"
            "每个镜头必须包含 episode/scene/shot/character/location/action/emotion/camera/duration 字段，"
            f"id 使用 'shot_{episode_number:03d}_{scene.scene_number:02d}_XX' 格式，scene_id 字段填 '{scene.id}'。"
            '输出必须是 {"shots": [...]} 结构。'
        )
        # ShotsFile 是包裹 shots 列表的结构化输出容器，这里只取出 shots 字段返回给调用方。
        shots_file = self.generate(user_prompt, sch.ShotsFile)
        return shots_file.shots

    def generate_dialogue(self, scene: sch.Scene, shots: list[sch.Shot]) -> list[sch.DialogueLine]:
        """根据镜头列表生成需要台词的对白，非每个镜头都需要对白，只为叙事必要的镜头生成。"""
        shot_lines = "\n".join(f"{sh.id}: {sh.action}（角色：{'、'.join(sh.character)}）" for sh in shots)
        user_prompt = (
            f"场次描述：{scene.description}\n出场角色 id：{'、'.join(scene.characters_present)}\n"
            f"镜头列表：\n{shot_lines}\n\n"
            "请为需要台词的镜头生成对白（DialogueLine），character_id 必须来自出场角色 id 列表，"
            "shot_id 必须对应上面某个镜头 id，order 从 1 开始递增，line_zh 为中文台词，line_en 给出对应英文翻译。"
            '不是每个镜头都需要台词，只为叙事必要的镜头生成。输出必须是 {"dialogue": [...]} 结构。'
        )
        # 借助 DialogueBatch 约束输出结构，再从中取出 dialogue 列表返回。
        batch = self.generate(user_prompt, DialogueBatch)
        return batch.dialogue
