"""把"一堆镜头素材"组装成 render_episode 需要的 EpisodeRenderConfig。

生产里每个镜头最终应该是一段视频（07_GENERATION/video），但视频生成是最贵最慢的一步；
在只有关键帧图片的阶段（或视频 provider 没配置时）也要能出成片做审片，所以这里支持
两种镜头素材：
- 视频文件：直接作为 timeline clip；
- 静态图：用 ffmpeg 生成"缓慢推近 + 静音音轨"的定长片段（Ken Burns 效果），再进 timeline。
对白（DialogueLine.line_zh）按镜头时长均分成字幕 cue，交给 09_POST/subtitle 烧录。

在流水线里的位置：上游（13_INFRA 的编排任务）拿到 07_GENERATION 的产物后调 render_from_media，
本模块负责"素材 -> 渲染配置"的翻译，真正的 ffmpeg 编排在 rendering/render.py。
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# 09_POST 目录名以数字开头，无法用普通 import 语句引用，只能把项目根加进 sys.path 后用 importlib 按字符串加载
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import importlib  # noqa: E402

_render = importlib.import_module("09_POST.rendering.render")
_concat = importlib.import_module("09_POST.editing.concat")

# 给上游一个稳定的引用名，避免调用方也要写一遍 importlib
EpisodeRenderConfig = _render.EpisodeRenderConfig
TimelineClip = _concat.TimelineClip


@dataclass
class ShotMedia:
    """一个镜头的素材描述：video_path 和 image_path 至少给一个，video 优先；duration_sec 决定静态图片段时长和字幕切分。"""

    shot_id: str
    duration_sec: float
    video_path: Optional[str] = None
    image_path: Optional[str] = None
    dialogue: list[str] | None = None  # 该镜头的台词（按顺序）


def still_to_clip(image_path: str | Path, duration_sec: float, output_path: str | Path, width: int = 1080, height: int = 1920, fps: int = 30) -> Path:
    """静态关键帧 -> 定长视频片段（轻微推近 + 静音音轨，保证 concat 时音视频流都在）。

    为什么要加静音音轨：concatenate_clips 的 filter 对每个输入都取 [i:a]，纯图片生成的视频没有音频流会直接报错。
    为什么用 zoompan 而不是直接静帧：几秒钟完全不动的画面在竖屏短剧里观感很"死"，轻微推近是最便宜的动态感。
    """
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
        "-loop",  # 单张图循环喂给编码器，配合 -t 得到定长视频
        "1",
        "-i",
        str(image_path),
        "-f",  # 第二路输入：lavfi 生成的静音立体声，采样率和 concat 阶段统一的 44100 对齐
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf",
        vf,
        "-t",
        f"{duration_sec:.3f}",
        "-c:v",
        "libx264",
        "-preset",  # 中间片段后面还要重编码，这里用 veryfast 省时间，画质损失可接受
        "veryfast",
        "-pix_fmt",  # yuv420p 是播放器/平台兼容性最好的像素格式
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",  # anullsrc 是无限长的，靠 -shortest 让输出在视频结束时停下
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg still_to_clip 失败：{result.stderr[-800:]}")
    return output_path


def build_dialogue_cues(shots: list[ShotMedia]) -> list[dict[str, Any]]:
    """把各镜头的台词按镜头时长均分成字幕 cue（cues_from_dialogue_json 接受的 dict 格式）。

    没有 TTS 时间戳可用，所以只能"均分"：一个镜头 N 句台词，每句占 duration/N 秒；
    end 减 0.05 秒是为了让相邻两条字幕之间有个极短的空隙，避免播放器把它们视为重叠。
    """
    cues: list[dict[str, Any]] = []
    cursor = 0.0  # 当前镜头在整集时间线上的起点
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
    """把 ShotMedia 列表翻译成 EpisodeRenderConfig：视频直接进时间线，静态图先转成片段；台词转成字幕 cue。

    静态图生成的中间片段放在 work_dir（默认 output_dir/_clips），不放系统临时目录，方便出问题时人工检查。
    """
    if not shots:
        raise ValueError("没有可用镜头素材")
    work = Path(work_dir or Path(output_dir) / "_clips")
    work.mkdir(parents=True, exist_ok=True)
    clips: list[TimelineClip] = []
    for shot in shots:
        if shot.video_path and Path(shot.video_path).exists():
            # 生成的视频可能比镜头设计时长长，用 trim_end_sec 截到设计时长，保证字幕时间轴对得上
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
        dialogue=dialogue or None,  # 空列表也当作 None，render_episode 就会整体跳过字幕步骤
        bgm_path=bgm_path,
    )


def render_from_media(episode_id: str, shots: list[ShotMedia], output_dir: str | Path, **kwargs) -> Path:
    """一步到位：素材列表 -> 渲染配置 -> Episode.mp4。kwargs 透传给 build_render_config。"""
    config = build_render_config(episode_id, shots, output_dir, **kwargs)
    return _render.render_episode(config)
