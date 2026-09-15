"""字幕：从流水线对白 JSON 或 Whisper ASR 结果生成 SRT，并用 ffmpeg 的 subtitles 滤镜（libass）硬烧进视频。

在 render_episode 里位于 9:16 适配之后。设计上把"写 SRT"和"硬烧"拆开：SRT 文件永远产出（平台上传/人工校对都要），
硬烧则依赖 ffmpeg 是否编译了 libass——没有时抛 SubtitleBurnUnavailable 让上游降级，而不是让整集渲染失败。
"""

from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

__all__ = [
    "SubtitleCue",
    "SubtitleBurnUnavailable",
    "burn_subtitles",
    "cues_from_dialogue_json",
    "cues_from_whisper_result",
    "ffmpeg_has_filter",
    "write_srt",
]


class SubtitleBurnUnavailable(RuntimeError):
    """本机 ffmpeg 没有编译 libass（subtitles 滤镜）：无法硬烧字幕，只能输出 SRT 旁路文件。
    Homebrew 默认的 ffmpeg 就是这样；Docker 镜像里 apt 装的 ffmpeg 带 libass。"""


@functools.cache
def ffmpeg_has_filter(name: str) -> bool:
    """检查本机 ffmpeg 是否有某个滤镜。结果缓存（一次进程内 ffmpeg 不会变），ffmpeg 不存在时返回 False 而不是抛错。"""
    try:
        result = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    # -filters 每行形如 " T.. subtitles  V->V  Render text subtitles..."，第二列才是滤镜名
    return any(line.split()[1:2] == [name] for line in result.stdout.splitlines() if line.strip())


PathLike = Union[str, Path]


@dataclass
class SubtitleCue:
    """一条字幕：SRT 序号、起止秒数、文本。"""

    index: int
    start_sec: float
    end_sec: float
    text: str


def _format_srt_timestamp(seconds: float) -> str:
    """秒 -> SRT 时间戳 "HH:MM:SS,mmm"（SRT 用逗号分隔毫秒）。负数按 0 处理。"""
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
    """从流水线的对白 JSON 格式构建字幕 cue，例如
    [{"text": "...", "start_sec": 0.0, "end_sec": 2.5}, ...]。字段名可通过 *_key 参数适配。
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
    """从 Whisper / faster-whisper 风格的 ASR 结果构建字幕 cue：
    {"segments": [{"start": 0.0, "end": 2.5, "text": "..."}], ...}。空文本段会被跳过，序号按实际保留的条数连续编号。
    """
    segments = whisper_result.get("segments", [])
    cues: List[SubtitleCue] = []
    for i, seg in enumerate(segments, start=1):
        text = str(seg["text"]).strip()
        if not text:
            continue
        cues.append(
            SubtitleCue(
                index=len(cues) + 1,  # 不用 i：跳过空段后序号仍要连续，否则部分播放器会报 SRT 格式错误
                start_sec=float(seg["start"]),
                end_sec=float(seg["end"]),
                text=text,
            )
        )
    return cues


def write_srt(cues: List[SubtitleCue], output_path: PathLike) -> Path:
    """把 cue 列表写成标准 SRT 文件（序号 / 时间轴 / 文本 / 空行）。"""
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
    """把路径转义成能放进 ffmpeg 滤镜参数的形式。

    滤镜串里冒号是参数分隔符、单引号是字符串定界符、反斜杠是转义符，路径里出现这些字符（如 Windows 盘符 C:）
    不转义会被解析成别的东西。
    """
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
    """用 ffmpeg 的 `subtitles` 滤镜（libass）把 SRT 硬烧进视频。

    ffmpeg 缺 subtitles 滤镜时抛 SubtitleBurnUnavailable（而非普通 RuntimeError），让 render_episode 能识别并降级。
    默认白字黑边、底部留 60px 边距，是竖屏短剧常见的样式；音频原样复制。
    """
    video_path = Path(video_path)
    srt_path = Path(srt_path)
    output_path = Path(output_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")
    if not ffmpeg_has_filter("subtitles"):
        raise SubtitleBurnUnavailable("ffmpeg 缺少 subtitles 滤镜（libass），无法硬烧字幕")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # force_style 用 ASS 样式语法：BorderStyle=1 是"描边 + 阴影"（不是背景框），Outline=2 描边宽度，颜色是 &HBBGGRR& 格式
    style = (
        f"FontSize={font_size},PrimaryColour=&H{_bgr_hex(font_color)}&,OutlineColour=&H{_bgr_hex(outline_color)}&,BorderStyle=1,Outline=2,MarginV={margin_v}"
    )
    subtitles_arg = f"subtitles=filename='{_escape_filter_path(srt_path)}':force_style='{style}'"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        subtitles_arg,
        "-c:a",
        "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg subtitle burn-in failed: {result.stderr}")
    return output_path


# 常用颜色名 -> ASS 的 BBGGRR 十六进制（注意是 BGR 顺序，所以 yellow 是 00FFFF 而不是 FFFF00）
_COLOR_NAMES = {
    "white": "FFFFFF",
    "black": "000000",
    "yellow": "00FFFF",
}


def _bgr_hex(color: str) -> str:
    """ASS/SSA 的颜色格式是 &HBBGGRR&；接受已知颜色名或 #RRGGBB 十六进制串，无法识别时回退白色。"""
    if color.lower() in _COLOR_NAMES:
        return _COLOR_NAMES[color.lower()]
    hex_str = color.lstrip("#")
    if len(hex_str) == 6:
        rr, gg, bb = hex_str[0:2], hex_str[2:4], hex_str[4:6]
        return f"{bb}{gg}{rr}".upper()  # RGB -> BGR 颠倒顺序
    return "FFFFFF"
