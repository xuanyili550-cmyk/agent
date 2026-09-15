"""视频质检：时长 / 分辨率边界（ffprobe）+ 黑帧、静帧检测（OpenCV 帧差采样）。

流水线位置：07_GENERATION 的视频 provider 产出 mp4 后、进入口型同步和后期之前的技术性闸门。
它不判断"画面好不好看"，只拦截明显的技术失败：生成中断导致的超短片段、分辨率不达标、
大面积黑场（模型失败常见输出）、整段几乎不动（图生视频退化成静态图）。
为什么用采样而不是逐帧：QC 要跑在每个镜头上，逐帧解码 + 灰度 + 差分对 1080p 视频太慢；
每 ``sample_stride`` 帧取一帧，对"比例"类指标已经足够准确。
"""

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
    """ffprobe 读出的视频元数据。"""

    duration_sec: float
    width: int
    height: int
    fps: float
    codec: str


@dataclass
class VideoQCResult:
    """一次视频质检的完整结果：元数据、四项检查各自的通过与否、以及用于定位问题的黑帧时间戳。"""

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
        """四项检查全部通过才算通过。"""
        return self.duration_ok and self.resolution_ok and self.black_frames_ok and self.still_frames_ok


class VideoQC:
    """视频质量检查：通过 ffprobe 检查时长 / 分辨率边界，再用 OpenCV 帧差采样做
    黑帧和静止帧检测。
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
        """阈值默认值的取法：

        - ``black_luma_threshold=16``：8 位灰度里 16 以下肉眼基本是纯黑，允许压缩噪声；
        - ``black_frame_max_ratio=0.10``：淡入淡出会带来少量黑帧，超过一成才算异常；
        - ``still_diff_threshold=2.0``：相邻采样帧平均灰度差 ≤2 视为没动（编码噪声约 1）；
        - ``still_frame_max_ratio=0.60``：短剧镜头允许人物静止对话，但超过六成不动多半是退化成静态图；
        - ``sample_stride=5``：25-30fps 下每 5 帧采一次约 0.2s 一帧，够判断比例又不至于太慢。
        """
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
        """调用 ffprobe 读取第一条视频流的时长、宽高、帧率、编码。"""
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

        # 优先取容器级 duration；某些封装只在流级别写了时长
        duration = float(info.get("format", {}).get("duration") or stream.get("duration") or 0.0)
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        codec = stream.get("codec_name", "unknown")

        fps = 0.0
        # ffprobe 的帧率是 "30000/1001" 这种分数字符串，需要自己算
        rate_str = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
        if "/" in rate_str:
            num, den = rate_str.split("/")
            den = float(den)
            fps = float(num) / den if den else 0.0
        else:
            fps = float(rate_str)

        return VideoProbe(duration_sec=duration, width=width, height=height, fps=fps, codec=codec)

    def _analyze_frames(self, video_path: Path):
        """按 sample_stride 抽帧，统计采样帧数、黑帧数、静帧数和黑帧时间戳。

        黑帧：灰度均值 ≤ black_luma_threshold；静帧：与上一采样帧的平均绝对差 ≤ still_diff_threshold。
        """
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
                        # fps 读不到时退化为帧号，至少还能定位
                        black_timestamps.append(frame_idx / fps if fps else float(frame_idx))
                    # shape 校验：极少数流中途改分辨率，absdiff 会直接崩
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
        """跑完整检查：probe 元数据 -> 时长 / 分辨率边界 -> 抽帧统计 -> 汇总成 VideoQCResult。"""
        path = Path(video_path)
        probe = self.probe(path)

        duration_ok = self.min_duration_sec <= probe.duration_sec <= self.max_duration_sec
        resolution_ok = probe.width >= self.min_width and probe.height >= self.min_height

        sampled, black_count, still_count, black_timestamps = self._analyze_frames(path)
        # 一帧都没采到（解码失败）按最坏情况处理：比例记 1.0，必定不通过
        black_ratio = (black_count / sampled) if sampled else 1.0
        # 静帧是相邻帧对比得出的，分母是 sampled - 1（至少为 1 防止除零）
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
