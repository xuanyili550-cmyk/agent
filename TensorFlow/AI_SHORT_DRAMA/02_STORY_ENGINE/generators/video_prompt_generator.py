"""视频 Prompt 生成器：把一个结构化镜头（Shot）渲染成文生视频模型可用的提示词。

与 image_prompt_generator 类似，这一步也是确定性的模板渲染（Jinja2），不调用 LLM，
负责把镜头的运镜方式、动作、情绪、时长等字段转换成视频生成模块
（07_GENERATION/video）能直接消费的 VideoPrompt 结构。
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

# 默认反向提示词：屏蔽视频生成中常见的画面抖动、人物崩坏等瑕疵。
DEFAULT_NEGATIVE_PROMPT = "画面抖动过度, 人物崩坏, 闪烁伪影, 镜头畸变"


def render_video_prompt_text(shot: sch.Shot, fps: int = 24) -> str:
    """用 Jinja2 模板把镜头的时长/动作/运镜/情绪等字段渲染成视频提示词文本。"""
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
    """基于镜头构建完整的 VideoPrompt 结构化对象，供视频生成模块直接使用。

    若传入了对应的 image_prompt（图生视频场景），会记录其 id 以便追溯首帧来源；
    提示词正文复用 render_video_prompt_text，再补上运镜描述、时长、目标模型等元数据。
    """
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
