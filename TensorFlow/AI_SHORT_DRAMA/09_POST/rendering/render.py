from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..editing.concat import TimelineClip, concatenate_clips
from ..music.music_mix import mix_bgm
from ..sfx.sfx_mix import SfxEvent, mix_sfx
from ..subtitle.subtitle import (
    cues_from_dialogue_json,
    cues_from_whisper_result,
    burn_subtitles,
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
    """Fits a video to a 9:16 vertical frame.

    mode="crop": scale to fully cover the frame, then center-crop the overflow (no bars,
    loses some edge content). mode="pad": scale to fit inside the frame, then pad the
    remainder with black bars (keeps full frame, adds letterboxing).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "crop":
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1"
        )
    elif mode == "pad":
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
        )
    else:
        raise ValueError(f"Unknown mode '{mode}', expected 'crop' or 'pad'")

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg vertical-fit failed: {result.stderr}")
    return output_path


@dataclass
class EpisodeRenderConfig:
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
    """Full post-production pipeline: concatenate shots -> fit 9:16 -> burn subtitles
    (if dialogue/ASR provided) -> mix BGM (if provided) -> mix SFX (if provided) ->
    Episode.mp4. Each step is skipped gracefully if its inputs aren't provided.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"render_{config.episode_id}_"))

    current = concatenate_clips(
        config.shots, work_dir / "01_timeline.mp4", width=config.width, height=config.height
    )
    current = fit_vertical_frame(
        current, work_dir / "02_vertical.mp4",
        width=config.width, height=config.height, mode=config.fit_mode,
    )

    if config.dialogue is not None or config.whisper_result is not None:
        cues = (
            cues_from_dialogue_json(config.dialogue)
            if config.dialogue is not None
            else cues_from_whisper_result(config.whisper_result)
        )
        srt_path = write_srt(cues, work_dir / "subtitles.srt")
        current = burn_subtitles(current, srt_path, work_dir / "03_subtitled.mp4")

    if config.bgm_path is not None:
        current = mix_bgm(
            current, config.bgm_path, work_dir / "04_bgm.mp4",
            bgm_volume_db=config.bgm_volume_db,
        )

    if config.sfx_events:
        current = mix_sfx(current, config.sfx_events, work_dir / "05_sfx.mp4")

    final_path = output_dir / "Episode.mp4"
    _run(["ffmpeg", "-y", "-i", str(current), "-c", "copy", str(final_path)])

    if not config.keep_intermediate:
        shutil.rmtree(work_dir, ignore_errors=True)

    return final_path


def _run(cmd: List[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg command failed: {' '.join(cmd)}\n{result.stderr}")
