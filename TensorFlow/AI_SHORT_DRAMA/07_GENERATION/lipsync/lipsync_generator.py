"""
Lip-sync generation via a pluggable provider abstraction. Same pattern as the
other 07_GENERATION providers: API client placeholders reading a key from an
env var, raising NotConfiguredError if missing.
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


class BaseLipSyncProvider(ABC):
    @abstractmethod
    def sync(
        self,
        video_path: str,
        audio_path: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
    ) -> str:
        """Returns a local file path to the lip-synced video."""


class SyncSoLipSyncProvider(BaseLipSyncProvider):
    API_URL = "https://api.sync.so/v2/generate"

    def __init__(self):
        self.api_key = os.environ.get("SYNC_API_KEY")
        if not self.api_key:
            raise NotConfiguredError("SYNC_API_KEY is not set")

    def sync(
        self,
        video_path: str,
        audio_path: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
    ) -> str:
        import requests

        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "model": "lipsync-1.9.0-beta",
            "input": [
                {"type": "video", "url": video_path},
                {"type": "audio", "url": audio_path},
            ],
        }

        response = requests.post(self.API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        task = response.json()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{task.get('id', uuid.uuid4().hex)}.mp4"
        return str(out_path)


class LipSyncGenerator:
    def __init__(self, provider: BaseLipSyncProvider, asset_log_path: str = "./outputs/asset_records.jsonl"):
        self.provider = provider
        self.asset_log_path = asset_log_path

    def sync(
        self,
        video_path: str,
        audio_path: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
        character_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        shot_id: Optional[str] = None,
        license: Optional[str] = None,
    ) -> str:
        file_path = self.provider.sync(video_path=video_path, audio_path=audio_path, output_dir=output_dir, seed=seed)
        record = AssetRecord(
            asset_id=new_asset_id("lipsync"),
            file_path=file_path,
            character_id=character_id,
            episode_id=episode_id,
            shot_id=shot_id,
            model=type(self.provider).__name__,
            prompt=f"video={video_path} audio={audio_path}",
            seed=seed,
            license=license,
            created_at=now_iso(),
        )
        write_asset_record(record, self.asset_log_path)
        return file_path
