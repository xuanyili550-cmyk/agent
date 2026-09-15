"""editing 子包：镜头拼接。导出 TimelineClip（时间线上的一个镜头）和两种拼接函数（重编码版 / 流复制快速版）。"""

from .concat import TimelineClip, concatenate_clips, concatenate_shots_stream_copy

__all__ = ["TimelineClip", "concatenate_clips", "concatenate_shots_stream_copy"]
