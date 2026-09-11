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


class PromptAgent(BaseAgent):
    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "prompt_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3):
        super().__init__(provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries)

    def generate_image_prompt(self, shot: sch.Shot) -> sch.ImagePrompt:
        user_prompt = (
            f"镜头信息：地点={shot.location}，角色={'、'.join(shot.character)}，动作={shot.action}，"
            f"情绪={shot.emotion.value}，景别={shot.camera.shot_size.value}，机位角度={shot.camera.angle.value}。\n"
            f"请为 shot_id='{shot.id}' 生成图像生成 prompt（竖屏 9:16，电影感短剧质感），id 使用 'img_{shot.id}'。"
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
