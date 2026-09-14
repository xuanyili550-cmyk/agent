"""把"一堆镜头素材"组装成 render_episode 需要的 EpisodeRenderConfig。

生产里每个镜头最终应该是一段视频（07_GENERATION/video），但视频生成是最贵最慢的一步；
在只有关键帧图片的阶段（或视频 provider 没配置时）也要能出成片做审片，所以这里支持
两种镜头素材：
- 视频文件：直接作为 timeline clip；
- 静态图：用 ffmpeg 生成"缓慢推近 + 静音音轨"的定长片段（Ken Burns 效果），再进 timeline。
对白（DialogueLine.line_zh）按镜头时长均分成字幕 cue，交给 09_POST/subtitle 烧录。
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import importlib  # noqa: E402

_render = importlib.import_module("09_POST.rendering.render")
_concat = importlib.import_module("09_POST.editing.concat")

EpisodeRenderConfig = _render.EpisodeRenderConfig
TimelineClip = _concat.TimelineClip


@dataclass
class ShotMedia:
    shot_id: str
    duration_sec: float
    video_path: Optional[str] = None
    image_path: Optional[str] = None
    dialogue: list[str] | None = None  # 该镜头的台词（按顺序）


def still_to_clip(image_path: str | Path, duration_sec: float, output_path: str | Path, width: int = 1080, height: int = 1920, fps: int = 30) -> Path:
    """静态关键帧 -> 定长视频片段（轻微推近 + 静音音轨，保证 concat 时音视频流都在）。"""
    image_path = Path(image_path)
    output_path = Path(output_path)
    if not image_path.exists():
        raise FileNotFoundError(f"关键帧不存在：{image_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, round(duration_sec * fps))
    # zoompan 做 1.0 -> 1.08 的缓慢推近；先把图缩放到 2 倍再 zoompan 避免抖动
    vf = (
        f"scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,crop={width * 2}:{height * 2},"
        f"zoompan=z='min(zoom+0.0008,1.08)':d={frames}:s={width}x{height}:fps={fps},format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf",
        vf,
        "-t",
        f"{duration_sec:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg still_to_clip 失败：{result.stderr[-800:]}")
    return output_path


def build_dialogue_cues(shots: list[ShotMedia]) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    cursor = 0.0
    for shot in shots:
        lines = [line for line in (shot.dialogue or []) if line.strip()]
        if lines:
            slot = shot.duration_sec / len(lines)
            for i, text in enumerate(lines):
                cues.append({"text": text, "start_sec": round(cursor + i * slot, 3), "end_sec": round(cursor + (i + 1) * slot - 0.05, 3)})
        cursor += shot.duration_sec
    return cues


def build_render_config(
    episode_id: str,
    shots: list[ShotMedia],
    output_dir: str | Path,
    *,
    width: int = 1080,
    height: int = 1920,
    bgm_path: Optional[str] = None,
    burn_subtitles: bool = True,
    work_dir: Optional[str | Path] = None,
) -> EpisodeRenderConfig:
    if not shots:
        raise ValueError("没有可用镜头素材")
    work = Path(work_dir or Path(output_dir) / "_clips")
    work.mkdir(parents=True, exist_ok=True)
    clips: list[TimelineClip] = []
    for shot in shots:
        if shot.video_path and Path(shot.video_path).exists():
            clips.append(TimelineClip(path=shot.video_path, shot_id=shot.shot_id, trim_end_sec=shot.duration_sec))
        elif shot.image_path:
            clip = still_to_clip(shot.image_path, shot.duration_sec, work / f"{shot.shot_id}.mp4", width=width, height=height)
            clips.append(TimelineClip(path=str(clip), shot_id=shot.shot_id))
        else:
            raise ValueError(f"镜头 {shot.shot_id} 既没有视频也没有关键帧")
    dialogue = build_dialogue_cues(shots) if burn_subtitles else None
    return EpisodeRenderConfig(
        episode_id=episode_id,
        output_dir=str(output_dir),
        shots=clips,
        width=width,
        height=height,
        dialogue=dialogue or None,
        bgm_path=bgm_path,
    )


def render_from_media(episode_id: str, shots: list[ShotMedia], output_dir: str | Path, **kwargs) -> Path:
    config = build_render_config(episode_id, shots, output_dir, **kwargs)
    return _render.render_episode(config)
