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

DEFAULT_STYLE_TAGS = ["cinematic", "vertical 9:16", "short-drama lighting", "photorealistic", "high contrast"]
DEFAULT_NEGATIVE_PROMPT = "低分辨率, 变形肢体, 多余手指, 水印, 卡通风格"


def render_image_prompt_text(shot: sch.Shot, style_tags: list[str] | None = None, aspect_ratio: str = "9:16") -> str:
    template = _env.get_template("image_prompt.jinja2")
    return template.render(
        location=shot.location,
        characters="、".join(shot.character),
        action=shot.action,
        emotion=shot.emotion.value,
        shot_size=shot.camera.shot_size.value,
        angle=shot.camera.angle.value,
        style_tags=style_tags or DEFAULT_STYLE_TAGS,
        aspect_ratio=aspect_ratio,
    ).strip()


def build_image_prompt(
    shot: sch.Shot,
    style_tags: list[str] | None = None,
    aspect_ratio: str = "9:16",
    seed: int | None = None,
) -> sch.ImagePrompt:
    return sch.ImagePrompt(
        id=f"img_{shot.id}",
        shot_id=shot.id,
        prompt_text=render_image_prompt_text(shot, style_tags=style_tags, aspect_ratio=aspect_ratio),
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        style_tags=style_tags or DEFAULT_STYLE_TAGS,
        aspect_ratio=aspect_ratio,
        reference_character_ids=shot.character,
        seed=seed,
    )
