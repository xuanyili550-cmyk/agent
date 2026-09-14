from __future__ import annotations

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

from pydantic import BaseModel, ConfigDict

import schemas as sch

from .base import PROMPTS_DIR, BaseAgent, LLMProvider
from .memory import ConversationMemory


class DialogueBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dialogue: list[sch.DialogueLine]


class StoryboardAgent(BaseAgent):
    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "storyboard_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3, memory: ConversationMemory | None = None):
        super().__init__(provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries, memory=memory)

    def generate_scene(self, episode: sch.Episode, scene_number: int, script: sch.Script) -> sch.Scene:
        user_prompt = (
            f"剧本内容：{script.content[:2000]}\n\n"
            f"请为 episode_id='{episode.id}' 生成第 {scene_number} 场（scene_number={scene_number}）的分场信息，"
            f"id 使用 'scene_{episode.episode_number:03d}_{scene_number:02d}'。shot_ids 和 dialogue 先留空列表，"
            "后续会单独生成分镜与对白。"
        )
        return self.generate(user_prompt, sch.Scene)

    def generate_shots(self, scene: sch.Scene, episode_number: int) -> list[sch.Shot]:
        user_prompt = (
            f"场次描述：{scene.description}\n出场角色：{'、'.join(scene.characters_present)}\n"
            f"请为 scene_id='{scene.id}'（episode={episode_number}, scene={scene.scene_number}）生成 2-4 个分镜镜头，"
            "每个镜头必须包含 episode/scene/shot/character/location/action/emotion/camera/duration 字段，"
            f"id 使用 'shot_{episode_number:03d}_{scene.scene_number:02d}_XX' 格式，scene_id 字段填 '{scene.id}'。"
            '输出必须是 {"shots": [...]} 结构。'
        )
        shots_file = self.generate(user_prompt, sch.ShotsFile)
        return shots_file.shots

    def generate_dialogue(self, scene: sch.Scene, shots: list[sch.Shot]) -> list[sch.DialogueLine]:
        shot_lines = "\n".join(f"{sh.id}: {sh.action}（角色：{'、'.join(sh.character)}）" for sh in shots)
        user_prompt = (
            f"场次描述：{scene.description}\n出场角色 id：{'、'.join(scene.characters_present)}\n"
            f"镜头列表：\n{shot_lines}\n\n"
            "请为需要台词的镜头生成对白（DialogueLine），character_id 必须来自出场角色 id 列表，"
            "shot_id 必须对应上面某个镜头 id，order 从 1 开始递增，line_zh 为中文台词，line_en 给出对应英文翻译。"
            '不是每个镜头都需要台词，只为叙事必要的镜头生成。输出必须是 {"dialogue": [...]} 结构。'
        )
        batch = self.generate(user_prompt, DialogueBatch)
        return batch.dialogue
