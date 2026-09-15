"""场景对齐质检：用 CLIP 文本-图像相似度（CLIPScore 风格）检查生成的画面是否符合场景描述。

流水线位置：07_GENERATION 生成关键帧图 / 视频片段后，与 03_STRUCTURED_DATA 的场景描述对比；
分数低于 QCThresholds.scene_min_similarity 时进入 RETRY（改写提示词）。
为什么用 CLIP：不需要训练、不需要标注，一个文本 + 一张图就能得到语义相似度，
对"该是夜晚小巷却生成了白天办公室"这类明显偏离足够敏感。
视频按等间隔抽几帧取平均，避免只看首帧导致中途跑偏的片段漏检。
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

from ..character.clip_backend import DEFAULT_CLIP_MODEL, CLIPBackend, CLIPModelUnavailableError

__all__ = ["SceneQC", "SceneAlignmentResult", "CLIPModelUnavailableError"]

PathLike = Union[str, Path]


@dataclass
class SceneAlignmentResult:
    """一次场景对齐打分的结果；``similarity`` 已缩放并 clamp 到 [0, 1]，``frames_sampled`` 记录参与平均的帧数。"""

    scene_description: str
    media_path: str
    similarity: float
    frames_sampled: int = 1


class SceneQC:
    """检查场景文本描述与其生成图像 / 视频之间的语义对齐度，使用 CLIP 文本-图像余弦相似度
    （CLIPScore 风格指标）。

    CLIP 的原始文本-图像余弦相似度对于对齐良好的样本通常落在 ~0.2-0.35 区间
    （不像图像-图像那样跨 [-1, 1]），因此这里用固定的 ``similarity_scale`` 放大后再 clamp 到 [0, 1]，
    而不是像 CharacterConsistencyChecker 里的图像-图像分数那样做线性重映射。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        device: Optional[str] = None,
        similarity_scale: float = 2.5,
    ):
        """记录模型名、设备和缩放系数；默认 2.5 让"对齐良好"的 0.2-0.35 落到 0.5-0.875，正好跨过 0.5 阈值。"""
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_scale = similarity_scale
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        """从进程级缓存拿 CLIP 模型并搬到目标设备；只在第一次调用时真正发生。"""
        if self._model is None:
            self._model, self._processor = CLIPBackend.get(self.model_name)
            self._model.to(self.device)

    @torch.no_grad()
    def _text_image_similarity(self, text: str, image: Image.Image) -> float:
        """一次前向同时得到文本和图像 embedding，各自 L2 归一化后点积即余弦相似度（原始值，未缩放）。"""
        self._ensure_loaded()
        inputs = self._processor(text=[text], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        text_emb = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        image_emb = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        return (text_emb @ image_emb.T).item()

    def _normalize(self, raw_similarity: float) -> float:
        """原始相似度乘以 similarity_scale 后 clamp 到 [0, 1]，得到可与阈值比较的 QC 分数。"""
        return max(0.0, min(1.0, raw_similarity * self.similarity_scale))

    def score_image(self, scene_description: str, image_path: PathLike) -> SceneAlignmentResult:
        """给单张图打场景对齐分。"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = Image.open(path).convert("RGB")
        raw = self._text_image_similarity(scene_description, image)
        return SceneAlignmentResult(
            scene_description=scene_description,
            media_path=str(path),
            similarity=self._normalize(raw),
            frames_sampled=1,
        )

    def score_video(
        self,
        scene_description: str,
        video_path: PathLike,
        num_frames: int = 3,
    ) -> SceneAlignmentResult:
        """通过 ffmpeg 从视频中等间隔抽取 ``num_frames`` 帧，对每帧计算与场景描述的
        文本-图像相似度后取平均。"""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        duration = self._probe_duration(path)
        similarities: List[float] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(num_frames):
                # (i + 0.5) / n：取每个等分区间的中点，避开首尾帧（常见淡入淡出黑场）
                timestamp = duration * (i + 0.5) / num_frames if duration > 0 else 0.0
                frame_path = Path(tmpdir) / f"frame_{i}.png"
                self._extract_frame(path, timestamp, frame_path)
                if frame_path.exists():
                    image = Image.open(frame_path).convert("RGB")
                    similarities.append(self._text_image_similarity(scene_description, image))
        if not similarities:
            raise RuntimeError(f"Could not extract any frames from {path}")
        avg_raw = sum(similarities) / len(similarities)
        return SceneAlignmentResult(
            scene_description=scene_description,
            media_path=str(path),
            similarity=self._normalize(avg_raw),
            frames_sampled=len(similarities),
        )

    @staticmethod
    def _probe_duration(video_path: Path) -> float:
        """用 ffprobe 读视频总时长（秒）；输出解析不了时返回 0.0，让调用方退化为只抽第 0 秒。"""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0

    @staticmethod
    def _extract_frame(video_path: Path, timestamp: float, out_path: Path) -> None:
        """用 ffmpeg 在指定时间点抽一帧存成 PNG；``-ss`` 放在 ``-i`` 前面是为了快速 seek 而不解码整段。"""
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
