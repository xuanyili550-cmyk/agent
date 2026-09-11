from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Union

__all__ = ["mix_bgm"]

PathLike = Union[str, Path]


def mix_bgm(
    video_path: PathLike,
    bgm_path: PathLike,
    output_path: PathLike,
    bgm_volume_db: float = -18.0,
    dialogue_volume_db: float = 0.0,
    fade_in_sec: float = 1.5,
    fade_out_sec: float = 2.0,
    loop_bgm: bool = True,
) -> Path:
    """Mixes a background music track under a video's existing dialogue/sfx audio.

    The BGM is attenuated (``bgm_volume_db``, typically negative) relative to dialogue,
    faded in/out, looped to cover the video duration if shorter, trimmed to match the
    video duration, then combined with the video's original audio via ffmpeg's amix.
    """
    video_path = Path(video_path)
    bgm_path = Path(bgm_path)
    output_path = Path(output_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not bgm_path.exists():
        raise FileNotFoundError(f"BGM file not found: {bgm_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = _probe_duration(video_path)
    fade_out_start = max(duration - fade_out_sec, 0.0)

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if loop_bgm:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-i", str(bgm_path)]

    filter_complex = (
        f"[0:a]volume={_db_to_linear(dialogue_volume_db):.6f}[dlg];"
        f"[1:a]volume={_db_to_linear(bgm_volume_db):.6f},"
        f"afade=t=in:st=0:d={fade_in_sec},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_sec}[bgm];"
        f"[dlg][bgm]amix=inputs=2:duration=first:dropout_transition=0[outa]"
    )

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[outa]",
        "-c:v", "copy", "-c:a", "aac", "-t", f"{duration:.3f}",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg BGM mix failed: {result.stderr}")
    return output_path


def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip() or 0.0)


def _db_to_linear(db: float) -> float:
    return 10 ** (db / 20.0)
