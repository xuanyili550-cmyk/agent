"""
Video generation via a pluggable provider abstraction. Concrete providers are
thin API client placeholders: they build a plausible request payload and read
their key from an env var, but this module makes no live network calls on
import, and raises NotConfiguredError at construction time if the key is
missing. Fill in actual endpoint/response handling once a real provider
contract is chosen.
"""
from __future__ import annotations

import os
import sys
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asset_registry import AssetRecord, new_asset_id, now_iso, write_asset_record  # noqa: E402
from errors import NotConfiguredError  # noqa: E402


class BaseVideoProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        duration_sec: float = 4.0,
        seed: Optional[int] = None,
        output_dir: str = "./outputs",
    ) -> str:
        """Returns a local file path to the generated video."""


class RunwayVideoProvider(BaseVideoProvider):
    API_URL = "https://api.dev.runwayml.com/v1/image_to_video"

    def __init__(self):
        self.api_key = os.environ.get("RUNWAY_API_KEY")
        if not self.api_key:
            raise NotConfiguredError("RUNWAY_API_KEY is not set")

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        duration_sec: float = 4.0,
        seed: Optional[int] = None,
        output_dir: str = "./outputs",
    ) -> str:
        import requests

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"promptText": prompt, "duration": duration_sec, "seed": seed}
        if image_path:
            payload["promptImage"] = image_path

        response = requests.post(self.API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        task = response.json()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{task.get('id', uuid.uuid4().hex)}.mp4"
        # NOTE: real integration needs to poll the task status endpoint and
        # download the finished asset; omitted here since this is a
        # structural placeholder with no live credentials to test against.
        return str(out_path)


class PikaVideoProvider(BaseVideoProvider):
    API_URL = "https://api.pika.art/v1/generate"

    def __init__(self):
        self.api_key = os.environ.get("PIKA_API_KEY")
        if not self.api_key:
            raise NotConfiguredError("PIKA_API_KEY is not set")

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        duration_sec: float = 4.0,
        seed: Optional[int] = None,
        output_dir: str = "./outputs",
    ) -> str:
        import requests

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"prompt": prompt, "duration": duration_sec, "seed": seed}
        if image_path:
            payload["image"] = image_path

        response = requests.post(self.API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        task = response.json()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{task.get('id', uuid.uuid4().hex)}.mp4"
        return str(out_path)


class MochiLocalVideoProvider(BaseVideoProvider):
    """Runs Genmo Mochi 1 (Apache-2.0, free incl. commercial use) locally via
    diffusers -- no API key, needs a GPU with a lot of VRAM and the weights
    cached locally (see 06_MODELS/video/registry.json: genmo/mochi-1-preview).
    image_path is accepted for interface parity with the API providers but is
    unused: Mochi here runs text-to-video, not image-to-video."""

    def __init__(self, model_id: str = "genmo/mochi-1-preview", device: str = "cuda", fps: int = 15):
        self.model_id = model_id
        self.device = device
        self.fps = fps
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import MochiPipeline

        pipe = MochiPipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        self._pipe = pipe
        return pipe

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        duration_sec: float = 4.0,
        seed: Optional[int] = None,
        output_dir: str = "./outputs",
    ) -> str:
        import torch
        from diffusers.utils import export_to_video

        pipe = self._load()
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        num_frames = max(1, round(duration_sec * self.fps))
        result = pipe(prompt=prompt, num_frames=num_frames, generator=generator)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{uuid.uuid4().hex}.mp4"
        export_to_video(result.frames[0], str(out_path), fps=self.fps)
        return str(out_path)


class VideoGenerator:
    def __init__(self, provider: BaseVideoProvider, asset_log_path: str = "./outputs/asset_records.jsonl"):
        self.provider = provider
        self.asset_log_path = asset_log_path

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        duration_sec: float = 4.0,
        seed: Optional[int] = None,
        output_dir: str = "./outputs",
        character_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        shot_id: Optional[str] = None,
        license: Optional[str] = None,
    ) -> str:
        file_path = self.provider.generate(
            prompt=prompt, image_path=image_path, duration_sec=duration_sec, seed=seed, output_dir=output_dir
        )
        record = AssetRecord(
            asset_id=new_asset_id("vid"),
            file_path=file_path,
            character_id=character_id,
            episode_id=episode_id,
            shot_id=shot_id,
            model=type(self.provider).__name__,
            prompt=prompt,
            seed=seed,
            license=license,
            created_at=now_iso(),
        )
        write_asset_record(record, self.asset_log_path)
        return file_path
