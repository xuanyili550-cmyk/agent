"""rendering 子包：整集渲染编排。导出 EpisodeRenderConfig（渲染参数）、fit_vertical_frame（9:16 适配）和 render_episode（总入口）。"""

from .render import EpisodeRenderConfig, fit_vertical_frame, render_episode

__all__ = ["EpisodeRenderConfig", "fit_vertical_frame", "render_episode"]
