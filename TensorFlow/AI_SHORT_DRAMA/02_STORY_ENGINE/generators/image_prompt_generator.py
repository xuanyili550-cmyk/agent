"""图像 Prompt 生成器：把一个结构化镜头（Shot）渲染成文生图模型可用的提示词。

这一步是确定性的模板渲染（Jinja2），不调用 LLM，目的是把 StoryboardAgent
生成的镜头结构化字段（地点、角色、动作、情绪、机位等）统一转换成图像生成模块
（07_GENERATION/image）能直接消费的 ImagePrompt 结构，保证提示词格式与风格标签、
反向提示词在所有镜头间保持一致。
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

from jinja2 import Environment, FileSystemLoader

import schemas as sch

_env = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
    trim_blocks=True,
    lstrip_blocks=True,
)

# 默认风格标签：竖屏短剧统一采用的电影感/高对比度写实风格，保证画面风格一致。
DEFAULT_STYLE_TAGS = ["cinematic", "vertical 9:16", "short-drama lighting", "photorealistic", "high contrast"]
# 默认反向提示词：屏蔽常见的图像生成瑕疵（低分辨率、变形肢体等）。
DEFAULT_NEGATIVE_PROMPT = "低分辨率, 变形肢体, 多余手指, 水印, 卡通风格"


def render_image_prompt_text(shot: sch.Shot, style_tags: list[str] | None = None, aspect_ratio: str = "9:16") -> str:
    """用 Jinja2 模板把镜头的地点/角色/动作/情绪/机位等字段渲染成图像提示词文本。"""
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
    """基于镜头构建完整的 ImagePrompt 结构化对象，供图像生成模块直接使用。

    提示词正文复用 render_image_prompt_text，再补上 id、反向提示词、参考角色 id、
    随机种子等图像生成模块需要的元数据字段。
    """
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
