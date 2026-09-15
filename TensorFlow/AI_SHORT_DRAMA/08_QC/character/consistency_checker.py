"""角色一致性质检：用 CLIP 图像 embedding 的余弦相似度判断生成图里的角色是否"还是那个人"。

流水线位置：07_GENERATION 生成关键帧图后，把它与角色定妆照（参考图）对比；
分数低于 QCThresholds.character_min_similarity 时，reports/qc_report.py 会给出 RETRY 决定，
触发三级重试阶梯的"改写提示词"一级。
为什么用 CLIP 而不是人脸识别：短剧角色可能是侧脸、远景、动画风格，人脸检测经常失败；
CLIP 图像 embedding 对整体外观（发型、服装、气质）同样敏感，作为粗筛闸门更稳。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from PIL import Image

from .clip_backend import DEFAULT_CLIP_MODEL, CLIPBackend, CLIPModelUnavailableError

__all__ = [
    "CharacterConsistencyChecker",
    "CharacterConsistencyResult",
    "CLIPModelUnavailableError",
    "DEFAULT_CLIP_MODEL",
]

PathLike = Union[str, Path]


@dataclass
class CharacterConsistencyResult:
    """一次角色一致性打分的结果；``similarity`` 已归一化到 [0, 1]。"""

    character_id: Optional[str]
    reference_image: str
    generated_image: str
    similarity: float


class CharacterConsistencyChecker:
    """用 CLIP 图像 embedding 余弦相似度给"生成的角色图与其参考图有多像"打分。

    原始余弦相似度在 [-1, 1]；为了当 QC 分数用，重新映射到 [0, 1]——
    因为无关图片之间的 CLIP 图像 embedding 很少会明显低于 0。
    """

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: Optional[str] = None):
        """记录模型名和设备（默认有 CUDA 就用 CUDA）；模型延迟到第一次打分时再从 CLIPBackend 取。"""
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        """从进程级缓存拿 CLIP 模型并搬到目标设备；只在第一次调用时真正发生。"""
        if self._model is None:
            self._model, self._processor = CLIPBackend.get(self.model_name)
            self._model.to(self.device)

    @staticmethod
    def _as_tensor(output) -> torch.Tensor:
        """get_image_features 在某些 transformers 版本返回裸 tensor，在另一些版本返回
        ModelOutput（带 .image_embeds 或 .pooler_output）；这里把两种情况统一成 tensor。"""
        if torch.is_tensor(output):
            return output
        for attr in ("image_embeds", "pooler_output"):
            value = getattr(output, attr, None)
            if value is not None:
                return value
        raise TypeError(f"Unexpected output type from CLIP get_image_features: {type(output)}")

    @torch.no_grad()
    def _embed_image(self, image_path: PathLike) -> torch.Tensor:
        """读图 -> CLIP 图像 embedding -> L2 归一化（归一化后点积即余弦相似度）。"""
        self._ensure_loaded()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = Image.open(path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self._as_tensor(self._model.get_image_features(**inputs))
        return features / features.norm(p=2, dim=-1, keepdim=True)

    def score(
        self,
        reference_image: PathLike,
        generated_image: PathLike,
        character_id: Optional[str] = None,
    ) -> CharacterConsistencyResult:
        """对比单张参考图与生成图，返回 [0, 1] 的相似度。"""
        ref_emb = self._embed_image(reference_image)
        gen_emb = self._embed_image(generated_image)
        cosine_sim = (ref_emb @ gen_emb.T).item()
        # (cos + 1) / 2 把 [-1, 1] 线性映射到 [0, 1]，再 clamp 防浮点误差越界
        similarity = max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))
        return CharacterConsistencyResult(
            character_id=character_id,
            reference_image=str(reference_image),
            generated_image=str(generated_image),
            similarity=similarity,
        )

    def score_against_references(
        self,
        reference_images: Iterable[PathLike],
        generated_image: PathLike,
        character_id: Optional[str] = None,
    ) -> CharacterConsistencyResult:
        """Best-of-N：对多张参考图（例如同一角色的不同姿势 / 角度）分别打分，返回最高的那个。"""
        results = [self.score(ref, generated_image, character_id) for ref in reference_images]
        if not results:
            raise ValueError("reference_images must contain at least one image")
        return max(results, key=lambda r: r.similarity)
