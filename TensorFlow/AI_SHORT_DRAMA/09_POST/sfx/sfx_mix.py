"""音效叠加：把若干一次性音效（开门声、手机震动等）在指定时间点叠到成片音轨上。

在 render_episode 里是最后一个音频处理步骤（BGM 之后）。和 music_mix 一样只重编码音频、视频流复制。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

__all__ = ["SfxEvent", "mix_sfx"]

PathLike = Union[str, Path]


@dataclass
class SfxEvent:
    """一个音效事件：音效文件路径、在成片时间线上的起始秒数、相对增益（dB，默认不变）。"""

    path: str
    start_sec: float
    volume_db: float = 0.0


def mix_sfx(
    video_path: PathLike,
    events: List[SfxEvent],
    output_path: PathLike,
) -> Path:
    """把一次性音效按固定时间点叠加到视频已有的音轨上。

    每个 SfxEvent 先用 ffmpeg 的 adelay 延迟到 start_sec，再调音量，最后所有音效和原始音频一起用 amix 合成。
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
        cmd += ["-i", str(event.path)]  # 每个音效作为一路独立输入，输入序号从 1 开始（0 是视频）

    filter_parts = []
    mix_labels = ["0:a"]  # 原始音轨永远是 amix 的第一路，配合 duration=first 决定输出时长
    for i, event in enumerate(events, start=1):
        delay_ms = max(int(round(event.start_sec * 1000)), 0)  # adelay 单位是毫秒，负数没意义
        gain = _db_to_linear(event.volume_db)
        # adelay 要为每个声道各给一个延迟值，"a|b" 对应左右声道，这里两边相同
        filter_parts.append(f"[{i}:a]volume={gain:.6f},adelay={delay_ms}|{delay_ms}[sfx{i}]")
        mix_labels.append(f"sfx{i}")

    mix_inputs = "".join(f"[{label}]" for label in mix_labels)
    # dropout_transition=0：音效很短，结束后不要让 amix 把其余音轨的音量"补"上来
    filter_complex = ";".join(filter_parts) + (f";{mix_inputs}amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0[outa]")

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v",
        "-map",
        "[outa]",
        "-c:v",  # 视频流原样复制
        "copy",
        "-c:a",
        "aac",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg SFX mix failed: {result.stderr}")
    return output_path


def _db_to_linear(db: float) -> float:
    """分贝 -> 线性增益倍数（ffmpeg volume 滤镜的参数形式）。"""
    return 10 ** (db / 20.0)
