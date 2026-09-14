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
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class AudioQCResult:
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
        return self.loudness_ok and self.silence_ok


class AudioQC:
    """Loudness and silence-gap checks via pydub (which shells out to ffmpeg for decoding).

    Loudness is measured as integrated dBFS (pydub's ``dBFS``), not full EBU R128 LUFS;
    good enough as a QC gate but not a mastering-grade loudness measurement.
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
        self.min_loudness_dbfs = min_loudness_dbfs
        self.max_loudness_dbfs = max_loudness_dbfs
        self.silence_threshold_dbfs = silence_threshold_dbfs
        self.min_silence_len_ms = min_silence_len_ms
        self.max_total_silence_ratio = max_total_silence_ratio
        self.max_single_silence_sec = max_single_silence_sec

    def check(self, audio_path: PathLike) -> AudioQCResult:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            audio = AudioSegment.from_file(path)
        except Exception as exc:
            raise RuntimeError(f"Could not decode audio file {path} (requires ffmpeg on PATH). Underlying error: {exc}") from exc

        duration_sec = len(audio) / 1000.0
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
