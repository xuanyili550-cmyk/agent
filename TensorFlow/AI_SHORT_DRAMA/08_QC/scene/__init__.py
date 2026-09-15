"""场景对齐质检子包：导出 SceneQC（文本-图像 CLIP 对齐检查器）和 SceneAlignmentResult（结果）。"""

from .scene_qc import SceneAlignmentResult, SceneQC

__all__ = ["SceneQC", "SceneAlignmentResult"]
