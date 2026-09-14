from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

__all__ = ["VideoQC", "VideoQCResult", "VideoProbe"]

PathLike = Union[str, Path]


@dataclass
class VideoProbe:
    duration_sec: float
    width: int
    height: int
    fps: float
    codec: str


@dataclass
class VideoQCResult:
    video_path: str
    probe: VideoProbe
    duration_ok: bool
    resolution_ok: bool
    black_frame_ratio: float
    still_frame_ratio: float
    black_frames_ok: bool
    still_frames_ok: bool
    sampled_frames: int
    black_frame_timestamps: List[float] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.duration_ok and self.resolution_ok and self.black_frames_ok and self.still_frames_ok


class VideoQC:
    """Video quality checks: duration/resolution bounds via ffprobe, plus black-frame and
    static/still-frame detection via OpenCV frame-difference sampling.
    """

    def __init__(
        self,
        min_duration_sec: float = 0.5,
        max_duration_sec: float = 600.0,
        min_width: int = 480,
        min_height: int = 480,
        black_luma_threshold: float = 16.0,
        black_frame_max_ratio: float = 0.10,
        still_diff_threshold: float = 2.0,
        still_frame_max_ratio: float = 0.60,
        sample_stride: int = 5,
    ):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.min_width = min_width
        self.min_height = min_height
        self.black_luma_threshold = black_luma_threshold
        self.black_frame_max_ratio = black_frame_max_ratio
        self.still_diff_threshold = still_diff_threshold
        self.still_frame_max_ratio = still_frame_max_ratio
        self.sample_stride = sample_stride

    def probe(self, video_path: PathLike) -> VideoProbe:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffprobe not found on PATH. Install ffmpeg (which provides ffprobe).") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffprobe failed on {path}: {exc.stderr}") from exc

        info = json.loads(result.stdout)
        video_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not video_streams:
            raise ValueError(f"No video stream found in {path}")
        stream = video_streams[0]

        duration = float(info.get("format", {}).get("duration") or stream.get("duration") or 0.0)
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        codec = stream.get("codec_name", "unknown")

        fps = 0.0
        rate_str = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
        if "/" in rate_str:
            num, den = rate_str.split("/")
            den = float(den)
            fps = float(num) / den if den else 0.0
        else:
            fps = float(rate_str)

        return VideoProbe(duration_sec=duration, width=width, height=height, fps=fps, codec=codec)

    def _analyze_frames(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        black_count = 0
        still_count = 0
        sampled = 0
        black_timestamps: List[float] = []
        prev_gray: Optional[np.ndarray] = None
        frame_idx = 0

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % self.sample_stride == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sampled += 1
                    mean_luma = float(gray.mean())
                    if mean_luma <= self.black_luma_threshold:
                        black_count += 1
                        black_timestamps.append(frame_idx / fps if fps else float(frame_idx))
                    if prev_gray is not None and prev_gray.shape == gray.shape:
                        diff = float(cv2.absdiff(prev_gray, gray).mean())
                        if diff <= self.still_diff_threshold:
                            still_count += 1
                    prev_gray = gray
                frame_idx += 1
        finally:
            cap.release()

        return sampled, black_count, still_count, black_timestamps

    def check(self, video_path: PathLike) -> VideoQCResult:
        path = Path(video_path)
        probe = self.probe(path)

        duration_ok = self.min_duration_sec <= probe.duration_sec <= self.max_duration_sec
        resolution_ok = probe.width >= self.min_width and probe.height >= self.min_height

        sampled, black_count, still_count, black_timestamps = self._analyze_frames(path)
        black_ratio = (black_count / sampled) if sampled else 1.0
        still_ratio = (still_count / max(sampled - 1, 1)) if sampled else 1.0

        return VideoQCResult(
            video_path=str(path),
            probe=probe,
            duration_ok=duration_ok,
            resolution_ok=resolution_ok,
            black_frame_ratio=black_ratio,
            still_frame_ratio=still_ratio,
            black_frames_ok=black_ratio <= self.black_frame_max_ratio,
            still_frames_ok=still_ratio <= self.still_frame_max_ratio,
            sampled_frames=sampled,
            black_frame_timestamps=black_timestamps,
        )
