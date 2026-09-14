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
    scene_description: str
    media_path: str
    similarity: float
    frames_sampled: int = 1


class SceneQC:
    """Checks semantic alignment between a scene's text description and its generated
    image or video, using CLIP text-image cosine similarity (CLIPScore-style metric).

    CLIP's raw text-image cosine similarity for well-aligned pairs typically sits in the
    ~0.2-0.35 range (not [-1, 1] like image-image), so it is rescaled with a fixed
    ``similarity_scale`` before being clamped to [0, 1] rather than being remapped like
    the image-image score in CharacterConsistencyChecker.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        device: Optional[str] = None,
        similarity_scale: float = 2.5,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_scale = similarity_scale
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model, self._processor = CLIPBackend.get(self.model_name)
            self._model.to(self.device)

    @torch.no_grad()
    def _text_image_similarity(self, text: str, image: Image.Image) -> float:
        self._ensure_loaded()
        inputs = self._processor(text=[text], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        text_emb = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        image_emb = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        return (text_emb @ image_emb.T).item()

    def _normalize(self, raw_similarity: float) -> float:
        return max(0.0, min(1.0, raw_similarity * self.similarity_scale))

    def score_image(self, scene_description: str, image_path: PathLike) -> SceneAlignmentResult:
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
        """Samples ``num_frames`` evenly-spaced frames from the video via ffmpeg and
        averages their text-image similarity against the scene description."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        duration = self._probe_duration(path)
        similarities: List[float] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(num_frames):
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
