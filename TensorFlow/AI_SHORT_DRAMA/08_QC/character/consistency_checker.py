from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from PIL import Image

from .clip_backend import CLIPBackend, CLIPModelUnavailableError, DEFAULT_CLIP_MODEL

__all__ = [
    "CharacterConsistencyChecker",
    "CharacterConsistencyResult",
    "CLIPModelUnavailableError",
    "DEFAULT_CLIP_MODEL",
]

PathLike = Union[str, Path]


@dataclass
class CharacterConsistencyResult:
    character_id: Optional[str]
    reference_image: str
    generated_image: str
    similarity: float


class CharacterConsistencyChecker:
    """Scores how visually consistent a generated character image is with its reference
    image(s), using CLIP image-embedding cosine similarity.

    Raw cosine similarity lies in [-1, 1]; it is remapped to [0, 1] for use as a QC score
    since CLIP image embeddings for unrelated images rarely go far below 0.
    """

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model, self._processor = CLIPBackend.get(self.model_name)
            self._model.to(self.device)

    @staticmethod
    def _as_tensor(output) -> torch.Tensor:
        """get_image_features returns a plain tensor on some transformers versions and a
        ModelOutput (with .image_embeds or .pooler_output) on others; normalize both."""
        if torch.is_tensor(output):
            return output
        for attr in ("image_embeds", "pooler_output"):
            value = getattr(output, attr, None)
            if value is not None:
                return value
        raise TypeError(f"Unexpected output type from CLIP get_image_features: {type(output)}")

    @torch.no_grad()
    def _embed_image(self, image_path: PathLike) -> torch.Tensor:
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
        ref_emb = self._embed_image(reference_image)
        gen_emb = self._embed_image(generated_image)
        cosine_sim = (ref_emb @ gen_emb.T).item()
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
        """Best-of-N: returns the highest similarity across multiple reference images
        (e.g. different poses/angles of the same character)."""
        results = [
            self.score(ref, generated_image, character_id) for ref in reference_images
        ]
        if not results:
            raise ValueError("reference_images must contain at least one image")
        return max(results, key=lambda r: r.similarity)
