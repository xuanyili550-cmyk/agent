"""镜头拼接：把多个 Shot 视频按顺序拼成一条 Scene/Episode 时间线。

这是 render_episode 的第一步。提供两条路径：
- concatenate_clips：先把每个片段重编码到统一分辨率/帧率/采样率再用 concat 滤镜拼，慢但对任意输入都可靠；
- concatenate_shots_stream_copy：用 concat demuxer 流复制，几乎不耗时，但要求所有输入编码参数完全一致。
生产默认走前者，因为不同镜头可能来自不同的视频生成模型，编码参数很难保证一致。
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

__all__ = ["TimelineClip", "concatenate_clips", "concatenate_shots_stream_copy"]

PathLike = Union[str, Path]


@dataclass
class TimelineClip:
    """时间线上的一个 Shot，可选入点/出点裁剪（秒）。trim_end_sec 是绝对出点，不是时长。"""

    path: str
    trim_start_sec: Optional[float] = None
    trim_end_sec: Optional[float] = None
    shot_id: Optional[str] = None


def _to_clip(item: Union[PathLike, TimelineClip]) -> TimelineClip:
    """把裸路径或 TimelineClip 统一成 TimelineClip，让调用方可以混着传。"""
    if isinstance(item, TimelineClip):
        return item
    return TimelineClip(path=str(item))


def _run(cmd: List[str]) -> None:
    """执行 ffmpeg 命令，非零退出码时把完整命令和 stderr 一起抛出，方便排查滤镜写错。"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg command failed: {' '.join(cmd)}\n{result.stderr}")


def concatenate_clips(
    clips: List[Union[PathLike, TimelineClip]],
    output_path: PathLike,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    audio_sample_rate: int = 44100,
) -> Path:
    """把多个 Shot 片段（可裁剪）拼成一条 Scene/Episode 时间线视频。

    在 concat 滤镜之前先把每个片段重编码到统一的分辨率/帧率/采样率，所以即使各镜头由不同模型、
    不同流水线生成、编码参数不一致也能正常工作——ffmpeg 的 concat *demuxer* 做不到这点，
    它要求所有输入流参数完全相同。
    """
    timeline = [_to_clip(c) for c in clips]
    if not timeline:
        raise ValueError("clips must contain at least one item")
    for clip in timeline:
        if not Path(clip.path).exists():
            raise FileNotFoundError(f"Shot clip not found: {clip.path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = ["ffmpeg", "-y"]
    for clip in timeline:
        # -ss / -t 放在对应 -i 之前是"输入侧裁剪"，ffmpeg 会直接 seek，比输出侧裁剪快
        if clip.trim_start_sec is not None:
            cmd += ["-ss", f"{clip.trim_start_sec:.3f}"]
        if clip.trim_end_sec is not None:
            duration = clip.trim_end_sec - (clip.trim_start_sec or 0.0)
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-i", str(clip.path)]

    filter_parts = []
    for i in range(len(timeline)):
        # 视频：等比缩放到能放进目标框（decrease），不足的边用黑边补齐并居中，setsar=1 防止像素宽高比不一致导致 concat 报错
        # 音频：重采样到统一采样率并强制立体声，否则 concat 滤镜会拒绝混合声道布局不同的流
        filter_parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[v{i}];"
            f"[{i}:a]aresample={audio_sample_rate},aformat=channel_layouts=stereo[a{i}];"
        )
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(timeline)))
    filter_complex = "".join(filter_parts) + f"{concat_inputs}concat=n={len(timeline)}:v=1:a=1[outv][outa]"

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-map",
        "[outa]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(output_path),
    ]
    _run(cmd)
    return output_path


def concatenate_shots_stream_copy(
    shot_paths: List[PathLike],
    output_path: PathLike,
) -> Path:
    """快速路径：通过 ffmpeg concat demuxer 做流复制拼接（不重编码）。

    只有当所有镜头已经是完全相同的编码/分辨率/帧率时才有效（例如全部由同一次生成流水线渲染）；
    否则这里不会报错，但 ffmpeg 可能拒绝输出或产出坏文件。除非确定输入是同质的，否则优先用 concatenate_clips()。
    """
    if not shot_paths:
        raise ValueError("shot_paths must contain at least one item")
    for p in shot_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Shot clip not found: {p}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # concat demuxer 需要一个列表文件，每行 file '<绝对路径>'；delete=False 是因为 ffmpeg 要在 with 块之外读它
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in shot_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
        list_path = f.name

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",  # 列表里是绝对路径，ffmpeg 默认视为"不安全"会拒绝，-safe 0 放行
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            str(output_path),
        ]
        _run(cmd)
    finally:
        Path(list_path).unlink(missing_ok=True)
    return output_path
