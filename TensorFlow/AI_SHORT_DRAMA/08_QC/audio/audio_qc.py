"""音频质检：响度范围 + 静音间隙检查（基于 pydub，解码走 ffmpeg）。

流水线位置：07_GENERATION 的 TTS 产出配音后、送口型同步和后期混音之前的技术性闸门。
拦截的是 TTS 常见失败模式：合成出几乎无声的文件（响度过低）、削波爆音（响度过高）、
中间出现长时间空白（模型在长句上"卡住"或截断）。
为什么用 dBFS 而不是 LUFS：pydub 的 dBFS 是简单 RMS 度量，不做响度加权，作为"有没有明显问题"
的门槛足够；最终成片的响度标准化放在 09_POST 做。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from pydub import AudioSegment
from pydub.silence import detect_silence

__all__ = ["AudioQC", "AudioQCResult", "SilenceSegment"]

PathLike = Union[str, Path]


@dataclass
class SilenceSegment:
    """一段静音区间（秒）。"""

    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        """区间长度（秒）。"""
        return self.end_sec - self.start_sec


@dataclass
class AudioQCResult:
    """一次音频质检的完整结果：时长、响度 / 峰值、静音区间列表以及各项是否通过。"""

    audio_path: str
    duration_sec: float
    loudness_dbfs: float
    peak_dbfs: float
    loudness_ok: bool
    silence_segments: List[SilenceSegment] = field(default_factory=list)
    total_silence_sec: float = 0.0
    silence_ok: bool = True

    @property
    def passed(self) -> bool:
        """响度和静音两项都通过才算通过。"""
        return self.loudness_ok and self.silence_ok


class AudioQC:
    """通过 pydub（解码时调用 ffmpeg）做响度和静音间隙检查。

    响度用的是积分 dBFS（pydub 的 ``dBFS``），不是完整的 EBU R128 LUFS；
    作为 QC 闸门够用，但不是母带级的响度测量。
    """

    def __init__(
        self,
        min_loudness_dbfs: float = -35.0,
        max_loudness_dbfs: float = -6.0,
        silence_threshold_dbfs: float = -45.0,
        min_silence_len_ms: int = 700,
        max_total_silence_ratio: float = 0.35,
        max_single_silence_sec: float = 3.0,
    ):
        """阈值默认值的取法：

        - 响度 [-35, -6] dBFS：正常对白 RMS 通常在 -25~-15，低于 -35 基本听不见，高于 -6 已接近削波；
        - ``silence_threshold_dbfs=-45``：低于 -45 视为静音，比响度下限再低 10 dB，避免把轻声说话当静音；
        - ``min_silence_len_ms=700``：短于 0.7s 的停顿是正常语句间隙，不计入静音；
        - ``max_total_silence_ratio=0.35``：一条配音超过 35% 是空白多半是 TTS 截断或卡住；
        - ``max_single_silence_sec=3.0``：单段空白超过 3s 在竖屏短剧里观众会以为播放器坏了。
        """
        self.min_loudness_dbfs = min_loudness_dbfs
        self.max_loudness_dbfs = max_loudness_dbfs
        self.silence_threshold_dbfs = silence_threshold_dbfs
        self.min_silence_len_ms = min_silence_len_ms
        self.max_total_silence_ratio = max_total_silence_ratio
        self.max_single_silence_sec = max_single_silence_sec

    def check(self, audio_path: PathLike) -> AudioQCResult:
        """解码音频，计算响度 / 峰值并检测静音区间，汇总成 AudioQCResult。"""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            audio = AudioSegment.from_file(path)
        except Exception as exc:
            raise RuntimeError(f"Could not decode audio file {path} (requires ffmpeg on PATH). Underlying error: {exc}") from exc

        duration_sec = len(audio) / 1000.0  # pydub 以毫秒计长度
        # 全零音频的 dBFS 是 -inf，会让后续比较和 JSON 序列化出问题，钳到 -120 这个"绝对安静"的有限值
        loudness_dbfs = audio.dBFS if audio.dBFS != float("-inf") else -120.0
        peak_dbfs = audio.max_dBFS if audio.max_dBFS != float("-inf") else -120.0
        loudness_ok = self.min_loudness_dbfs <= loudness_dbfs <= self.max_loudness_dbfs

        raw_silences = detect_silence(
            audio,
            min_silence_len=self.min_silence_len_ms,
            silence_thresh=self.silence_threshold_dbfs,
        )
        segments = [SilenceSegment(start_sec=start_ms / 1000.0, end_sec=end_ms / 1000.0) for start_ms, end_ms in raw_silences]
        total_silence_sec = sum(s.duration_sec for s in segments)
        # 零时长文件按最坏情况：静音比例记 1.0，必定不通过
        silence_ratio = (total_silence_sec / duration_sec) if duration_sec > 0 else 1.0
        longest_silence = max((s.duration_sec for s in segments), default=0.0)

        silence_ok = silence_ratio <= self.max_total_silence_ratio and longest_silence <= self.max_single_silence_sec

        return AudioQCResult(
            audio_path=str(path),
            duration_sec=duration_sec,
            loudness_dbfs=loudness_dbfs,
            peak_dbfs=peak_dbfs,
            loudness_ok=loudness_ok,
            silence_segments=segments,
            total_silence_sec=total_silence_sec,
            silence_ok=silence_ok,
        )
