from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

__all__ = ["SfxEvent", "mix_sfx"]

PathLike = Union[str, Path]


@dataclass
class SfxEvent:
    path: str
    start_sec: float
    volume_db: float = 0.0


def mix_sfx(
    video_path: PathLike,
    events: List[SfxEvent],
    output_path: PathLike,
) -> Path:
    """Overlays one-shot sound effects onto a video's existing audio at fixed timestamps.

    Each SfxEvent is delayed to its start_sec (via ffmpeg's adelay) and volume-adjusted,
    then all events plus the original audio are combined with amix.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not events:
        raise ValueError("events must contain at least one SfxEvent")
    for event in events:
        if not Path(event.path).exists():
            raise FileNotFoundError(f"SFX file not found: {event.path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    for event in events:
        cmd += ["-i", str(event.path)]

    filter_parts = []
    mix_labels = ["0:a"]
    for i, event in enumerate(events, start=1):
        delay_ms = max(int(round(event.start_sec * 1000)), 0)
        gain = _db_to_linear(event.volume_db)
        filter_parts.append(
            f"[{i}:a]volume={gain:.6f},adelay={delay_ms}|{delay_ms}[sfx{i}]"
        )
        mix_labels.append(f"sfx{i}")

    mix_inputs = "".join(f"[{label}]" for label in mix_labels)
    filter_complex = ";".join(filter_parts) + (
        f";{mix_inputs}amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0[outa]"
    )

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[outa]",
        "-c:v", "copy", "-c:a", "aac",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg SFX mix failed: {result.stderr}")
    return output_path


def _db_to_linear(db: float) -> float:
    return 10 ** (db / 20.0)
