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


class PromptAgent(BaseAgent):
    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "prompt_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3, memory: ConversationMemory | None = None):
        super().__init__(provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries, memory=memory)

    @staticmethod
    def _character_sheet(shot: sch.Shot, characters: list[sch.Character] | None) -> str:
        if not characters:
            return ""
        present = [c for c in characters if c.id in shot.character]
        if not present:
            return ""
        # 把角色外貌写进 prompt 是跨镜头长相一致的第一道保险（第二道是参考图/LoRA）
        return "角色外貌设定（必须逐字保留关键特征）：\n" + "\n".join(f"- {c.name}（{c.id}）：{c.appearance}" for c in present) + "\n"

    def generate_image_prompt(self, shot: sch.Shot, characters: list[sch.Character] | None = None) -> sch.ImagePrompt:
        user_prompt = (
            f"镜头信息：地点={shot.location}，角色={'、'.join(shot.character)}，动作={shot.action}，"
            f"情绪={shot.emotion.value}，景别={shot.camera.shot_size.value}，机位角度={shot.camera.angle.value}。\n"
            f"{self._character_sheet(shot, characters)}"
            f"请为 shot_id='{shot.id}' 生成图像生成 prompt（竖屏 9:16，电影感短剧质感），id 使用 'img_{shot.id}'，"
            f"reference_character_ids 填 {list(shot.character)}。"
        )
        return self.generate(user_prompt, sch.ImagePrompt)

    def generate_video_prompt(self, shot: sch.Shot, image_prompt: sch.ImagePrompt) -> sch.VideoPrompt:
        user_prompt = (
            f"关键帧图像 prompt：{image_prompt.prompt_text}\n"
            f"镜头运动：{shot.camera.movement.value}，时长：{shot.duration} 秒，情绪：{shot.emotion.value}。\n"
            f"请为 shot_id='{shot.id}' 生成视频生成 prompt，id 使用 'vid_{shot.id}'，"
            f"image_prompt_id 填 '{image_prompt.id}'，camera_movement 使用 '{shot.camera.movement.value}'，"
            f"duration 使用 {shot.duration}。"
        )
        return self.generate(user_prompt, sch.VideoPrompt)

    def rewrite_image_prompt(self, image_prompt: sch.ImagePrompt, failure_reasons: list[str], attempt: int) -> sch.ImagePrompt:
        """三级重试的第二级：生成结果没过 QC 或模型报错时，让 LLM 换个说法重写 prompt。

        要求保留 shot_id / id / reference_character_ids 不变——重写的是措辞，不是镜头本身。
        """
        reasons = "\n".join(f"- {r}" for r in failure_reasons) or "- 生成失败，原因未知"
        user_prompt = (
            f"下面这条图像 prompt 已经连续 {attempt} 次生成失败或未通过质检：\n{image_prompt.prompt_text}\n"
            f"negative_prompt：{image_prompt.negative_prompt}\n失败原因：\n{reasons}\n\n"
            "请换一种表达方式重写 prompt_text：调整措辞、补充缺失的视觉细节、去掉可能触发模型安全过滤或导致画面崩坏的描述，"
            "把角色外貌关键特征写得更明确。保持镜头内容不变。"
            f"id 保持 '{image_prompt.id}'，shot_id 保持 '{image_prompt.shot_id}'，"
            f"reference_character_ids 保持 {image_prompt.reference_character_ids}，aspect_ratio 保持 '{image_prompt.aspect_ratio}'。"
        )
        rewritten = self.generate(user_prompt, sch.ImagePrompt)
        return rewritten.model_copy(
            update={
                "id": image_prompt.id,
                "shot_id": image_prompt.shot_id,
                "reference_character_ids": image_prompt.reference_character_ids,
                "aspect_ratio": image_prompt.aspect_ratio,
                "seed": None,
            }
        )
