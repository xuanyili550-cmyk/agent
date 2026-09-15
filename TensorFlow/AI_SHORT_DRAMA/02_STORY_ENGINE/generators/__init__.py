"""生成器包：统一导出图像 Prompt 与视频 Prompt 的构建函数。

这个包不依赖 LLM，只做「结构化镜头数据 -> 文生图/文生视频提示词文本」的确定性渲染
（通过 Jinja2 模板），属于流水线里图像生成、视频生成模块的直接上游。
"""

from .image_prompt_generator import build_image_prompt, render_image_prompt_text
from .video_prompt_generator import build_video_prompt, render_video_prompt_text

__all__ = [
    "build_image_prompt",
    "render_image_prompt_text",
    "build_video_prompt",
    "render_video_prompt_text",
]
