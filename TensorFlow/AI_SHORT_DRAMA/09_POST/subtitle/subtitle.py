from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

__all__ = [
    "SubtitleCue",
    "burn_subtitles",
    "cues_from_dialogue_json",
    "cues_from_whisper_result",
    "write_srt",
]

PathLike = Union[str, Path]


@dataclass
class SubtitleCue:
    index: int
    start_sec: float
    end_sec: float
    text: str


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def cues_from_dialogue_json(
    dialogue: List[Dict[str, Any]],
    text_key: str = "text",
    start_key: str = "start_sec",
    end_key: str = "end_sec",
) -> List[SubtitleCue]:
    """Builds subtitle cues from the pipeline's Dialogue JSON format, e.g.
    [{"text": "...", "start_sec": 0.0, "end_sec": 2.5}, ...].
    """
    cues: List[SubtitleCue] = []
    for i, line in enumerate(dialogue, start=1):
        cues.append(
            SubtitleCue(
                index=i,
                start_sec=float(line[start_key]),
                end_sec=float(line[end_key]),
                text=str(line[text_key]).strip(),
            )
        )
    return cues


def cues_from_whisper_result(whisper_result: Dict[str, Any]) -> List[SubtitleCue]:
    """Builds subtitle cues from a Whisper/faster-whisper style ASR result:
    {"segments": [{"start": 0.0, "end": 2.5, "text": "..."}], ...}.
    """
    segments = whisper_result.get("segments", [])
    cues: List[SubtitleCue] = []
    for i, seg in enumerate(segments, start=1):
        text = str(seg["text"]).strip()
        if not text:
            continue
        cues.append(
            SubtitleCue(
                index=len(cues) + 1,
                start_sec=float(seg["start"]),
                end_sec=float(seg["end"]),
                text=text,
            )
        )
    return cues


def write_srt(cues: List[SubtitleCue], output_path: PathLike) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cue in cues:
        lines.append(str(cue.index))
        lines.append(f"{_format_srt_timestamp(cue.start_sec)} --> {_format_srt_timestamp(cue.end_sec)}")
        lines.append(cue.text)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _escape_filter_path(path: Path) -> str:
    s = str(path.resolve())
    s = s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return s


def burn_subtitles(
    video_path: PathLike,
    srt_path: PathLike,
    output_path: PathLike,
    font_size: int = 20,
    font_color: str = "white",
    outline_color: str = "black",
    margin_v: int = 60,
) -> Path:
    """Hard-burns an SRT file into a video via ffmpeg's `subtitles` filter (libass)."""
    video_path = Path(video_path)
    srt_path = Path(srt_path)
    output_path = Path(output_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    style = (
        f"FontSize={font_size},PrimaryColour=&H{_bgr_hex(font_color)}&,"
        f"OutlineColour=&H{_bgr_hex(outline_color)}&,BorderStyle=1,Outline=2,MarginV={margin_v}"
    )
    subtitles_arg = f"subtitles=filename='{_escape_filter_path(srt_path)}':force_style='{style}'"

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", subtitles_arg,
        "-c:a", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg subtitle burn-in failed: {result.stderr}")
    return output_path


_COLOR_NAMES = {
    "white": "FFFFFF",
    "black": "000000",
    "yellow": "00FFFF",
}


def _bgr_hex(color: str) -> str:
    """ASS/SSA colors are &HBBGGRR&; accepts a known name or a #RRGGBB hex string."""
    if color.lower() in _COLOR_NAMES:
        return _COLOR_NAMES[color.lower()]
    hex_str = color.lstrip("#")
    if len(hex_str) == 6:
        rr, gg, bb = hex_str[0:2], hex_str[2:4], hex_str[4:6]
        return f"{bb}{gg}{rr}".upper()
    return "FFFFFF"
