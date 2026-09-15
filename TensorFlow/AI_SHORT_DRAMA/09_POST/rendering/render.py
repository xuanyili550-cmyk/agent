"""整集渲染编排：拼接镜头 -> 适配 9:16 竖屏 -> 硬烧字幕 -> 混 BGM -> 叠音效 -> Episode.mp4。

这是 09_POST 的总入口（assemble.py 只是往这里喂配置）。每一步都是"有输入才做、没有就跳过"，
所以只有镜头视频的最小配置也能出片；每一步的产物都落在独立的临时工作目录里，便于 keep_intermediate 调试。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..editing.concat import TimelineClip, concatenate_clips
from ..music.music_mix import mix_bgm
from ..sfx.sfx_mix import SfxEvent, mix_sfx
from ..subtitle.subtitle import (
    SubtitleBurnUnavailable,
    burn_subtitles,
    cues_from_dialogue_json,
    cues_from_whisper_result,
    write_srt,
)

__all__ = ["EpisodeRenderConfig", "fit_vertical_frame", "render_episode"]

PathLike = Union[str, Path]


def fit_vertical_frame(
    input_path: PathLike,
    output_path: PathLike,
    width: int = 1080,
    height: int = 1920,
    mode: str = "crop",
) -> Path:
    """把视频适配到 9:16 竖屏画框。

    mode="crop"：缩放到完全覆盖画框，再居中裁掉溢出部分（无黑边，但会丢一点边缘内容）。
    mode="pad"：缩放到完整放进画框，剩余部分补黑边（保留全部画面，但有上下/左右黑边）。
    默认 crop，因为竖屏短剧平台上黑边非常影响观感。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "crop":
        # increase：等比放大到至少覆盖目标框，然后 crop 到精确尺寸
        vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1"
    elif mode == "pad":
        # decrease：等比缩小到能放进目标框，然后用黑色 pad 补齐并居中
        vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    else:
        raise ValueError(f"Unknown mode '{mode}', expected 'crop' or 'pad'")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",  # 只改画面，音频原样复制
        "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg vertical-fit failed: {result.stderr}")
    return output_path


@dataclass
class EpisodeRenderConfig:
    """一集的渲染参数。

    shots 是必填的镜头列表；dialogue（流水线对白 JSON）和 whisper_result（ASR 结果）二选一提供字幕来源，
    dialogue 优先；bgm_path / sfx_events 可选；keep_intermediate=True 时保留临时工作目录用于排查。
    """

    episode_id: str
    output_dir: str
    shots: List[Union[str, TimelineClip]]
    width: int = 1080
    height: int = 1920
    fit_mode: str = "crop"
    dialogue: Optional[List[Dict[str, Any]]] = None
    whisper_result: Optional[Dict[str, Any]] = None
    bgm_path: Optional[str] = None
    bgm_volume_db: float = -18.0
    sfx_events: Optional[List[SfxEvent]] = None
    keep_intermediate: bool = False


def render_episode(config: EpisodeRenderConfig) -> Path:
    """完整后期流水线：拼接镜头 -> 适配 9:16 -> 硬烧字幕（提供了对白/ASR 时）-> 混 BGM（提供了时）
    -> 叠音效（提供了时）-> Episode.mp4。任何一步缺少输入都会被静默跳过。
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"render_{config.episode_id}_"))

    # current 始终指向"当前最新的中间产物"，每一步以它为输入、产出下一个编号文件
    current = concatenate_clips(config.shots, work_dir / "01_timeline.mp4", width=config.width, height=config.height)
    current = fit_vertical_frame(
        current,
        work_dir / "02_vertical.mp4",
        width=config.width,
        height=config.height,
        mode=config.fit_mode,
    )

    if config.dialogue is not None or config.whisper_result is not None:
        cues = cues_from_dialogue_json(config.dialogue) if config.dialogue is not None else cues_from_whisper_result(config.whisper_result)
        srt_path = write_srt(cues, work_dir / "subtitles.srt")
        # SRT 旁路文件永远保留在成片旁边：平台上传时可作为字幕文件提交，也方便人工校对
        shutil.copyfile(srt_path, output_dir / "Episode.srt")
        try:
            current = burn_subtitles(current, srt_path, work_dir / "03_subtitled.mp4")
        except SubtitleBurnUnavailable as exc:
            # 本机 ffmpeg 没有 libass：不硬烧，成片照出，字幕以 SRT 旁路文件形式交付
            print(f"[render] 跳过字幕硬烧：{exc}")

    if config.bgm_path is not None:
        current = mix_bgm(
            current,
            config.bgm_path,
            work_dir / "04_bgm.mp4",
            bgm_volume_db=config.bgm_volume_db,
        )

    if config.sfx_events:
        current = mix_sfx(current, config.sfx_events, work_dir / "05_sfx.mp4")

    # 最后一步用 -c copy 把中间产物"搬"到输出目录：不重编码，只是重新封装，顺便让 moov 等元数据规整
    final_path = output_dir / "Episode.mp4"
    _run(["ffmpeg", "-y", "-i", str(current), "-c", "copy", str(final_path)])

    if not config.keep_intermediate:
        shutil.rmtree(work_dir, ignore_errors=True)

    return final_path


def _run(cmd: List[str]) -> None:
    """执行 ffmpeg 命令，失败时带完整命令和 stderr 抛错。"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg command failed: {' '.join(cmd)}\n{result.stderr}")
