from __future__ import annotations

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

from jinja2 import Environment, FileSystemLoader

import schemas as sch

_env = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
    trim_blocks=True,
    lstrip_blocks=True,
)

DEFAULT_NEGATIVE_PROMPT = "画面抖动过度, 人物崩坏, 闪烁伪影, 镜头畸变"


def render_video_prompt_text(shot: sch.Shot, fps: int = 24) -> str:
    template = _env.get_template("video_prompt.jinja2")
    return template.render(
        duration=shot.duration,
        action=shot.action,
        movement=shot.camera.movement.value,
        emotion=shot.emotion.value,
        fps=fps,
    ).strip()


def build_video_prompt(
    shot: sch.Shot,
    image_prompt: sch.ImagePrompt | None = None,
    fps: int = 24,
    model_target: str = "generic",
) -> sch.VideoPrompt:
    return sch.VideoPrompt(
        id=f"vid_{shot.id}",
        shot_id=shot.id,
        image_prompt_id=image_prompt.id if image_prompt else None,
        prompt_text=render_video_prompt_text(shot, fps=fps),
        motion_description=f"{shot.camera.movement.value} 运镜，{shot.duration} 秒内完成动作弧线",
        camera_movement=shot.camera.movement,
        duration=shot.duration,
        fps=fps,
        model_target=model_target,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    )
