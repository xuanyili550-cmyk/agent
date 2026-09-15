"""背景音乐混音：把一条 BGM 压低音量、加淡入淡出后垫在成片已有的对白/音效音轨下面。

在 render_episode 里位于字幕硬烧之后、音效叠加之前。视频流原样复制（-c:v copy），只重编码音频，
所以这一步很快，也不会二次损失画质。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Union

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
    """把背景音乐混到视频已有的对白/音效音轨下面。

    BGM 相对对白衰减 ``bgm_volume_db``（通常是负数，默认 -18 dB 让人声始终清晰），做淡入淡出，
    比视频短时循环补齐，再截到视频时长，最后用 ffmpeg 的 amix 与原始音频合成。
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
    fade_out_start = max(duration - fade_out_sec, 0.0)  # 淡出要在视频结束前 fade_out_sec 秒开始

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if loop_bgm:
        cmd += ["-stream_loop", "-1"]  # -1 = 无限循环，放在 BGM 的 -i 之前才对该输入生效；长度靠后面的 -t 截断
    cmd += ["-i", str(bgm_path)]

    # amix 的 duration=first 让输出时长跟随第一路（对白）；dropout_transition=0 避免某路先结束时 amix 自动拉高剩余音量
    filter_complex = (
        f"[0:a]volume={_db_to_linear(dialogue_volume_db):.6f}[dlg];"
        f"[1:a]volume={_db_to_linear(bgm_volume_db):.6f},"
        f"afade=t=in:st=0:d={fade_in_sec},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_sec}[bgm];"
        f"[dlg][bgm]amix=inputs=2:duration=first:dropout_transition=0[outa]"
    )

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v",
        "-map",
        "[outa]",
        "-c:v",  # 视频不动，只重编码音频
        "copy",
        "-c:a",
        "aac",
        "-t",  # 双保险：即使 BGM 无限循环，输出也严格截到视频时长
        f"{duration:.3f}",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg BGM mix failed: {result.stderr}")
    return output_path


def _probe_duration(path: Path) -> float:
    """用 ffprobe 读媒体文件总时长（秒）；输出为空时返回 0.0。"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",  # 只打印数值本身，不带 key 和 [FORMAT] 包裹，方便直接 float()
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip() or 0.0)


def _db_to_linear(db: float) -> float:
    """分贝 -> 线性增益（ffmpeg volume 滤镜要的是倍数）：0 dB = 1.0，-6 dB ≈ 0.5，-18 dB ≈ 0.126。"""
    return 10 ** (db / 20.0)
