"""
Image generation. DiffusersImageGenerator is a real local diffusers pipeline
wrapper (needs GPU + downloaded weights to actually run). DummyImageGenerator
draws a labelled placeholder PNG instead, so pipelines can be smoke-tested
offline with no model weights.
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04_DATASET" / "images"))
from generate_placeholder_images import COLORS  # noqa: E402


class BaseImageGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        lora_path: Optional[str] = None,
    ) -> Image.Image:
        ...


class DiffusersImageGenerator(BaseImageGenerator):
    """Wraps diffusers.AutoPipelineForText2Image. Requires torch + a GPU (or
    slow CPU inference) and downloaded/cached model weights -- not exercised
    by the offline demo."""

    def __init__(self, model_id_or_path: str, device: str = "cuda", dtype: str = "float16"):
        self.model_id_or_path = model_id_or_path
        self.device = device
        self.dtype = dtype
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import AutoPipelineForText2Image

        torch_dtype = getattr(torch, self.dtype)
        pipe = AutoPipelineForText2Image.from_pretrained(self.model_id_or_path, torch_dtype=torch_dtype)
        pipe = pipe.to(self.device)
        self._pipe = pipe
        return pipe

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        lora_path: Optional[str] = None,
    ) -> Image.Image:
        import torch

        pipe = self._load()
        if lora_path:
            pipe.load_lora_weights(lora_path)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]


class DummyImageGenerator(BaseImageGenerator):
    """No model weights, no GPU. Draws a solid-color labelled placeholder so
    the rest of the pipeline (asset records, file layout) can be exercised
    offline."""

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 1,
        guidance_scale: float = 0.0,
        lora_path: Optional[str] = None,
    ) -> Image.Image:
        color = COLORS[(seed or 0) % len(COLORS)]
        img = Image.new("RGB", (width, height), color=color)
        draw = ImageDraw.Draw(img)
        draw.multiline_text((16, 16), prompt[:200], fill=(255, 255, 255))
        return img
