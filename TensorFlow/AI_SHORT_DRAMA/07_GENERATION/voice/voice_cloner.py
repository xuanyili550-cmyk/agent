"""
Voice cloning via a pluggable provider abstraction. Same pattern as the other
07_GENERATION providers: API client placeholders reading a key from an env
var, raising NotConfiguredError if missing.
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asset_registry import AssetRecord, new_asset_id, now_iso, write_asset_record  # noqa: E402
from errors import NotConfiguredError  # noqa: E402


class BaseVoiceCloneProvider(ABC):
    @abstractmethod
    def clone_voice(self, name: str, reference_audio_paths: list[str]) -> str:
        """Returns a provider-side voice_id that can be passed to a TTS provider."""


class ElevenLabsVoiceCloneProvider(BaseVoiceCloneProvider):
    API_URL = "https://api.elevenlabs.io/v1/voices/add"

    def __init__(self):
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise NotConfiguredError("ELEVENLABS_API_KEY is not set")

    def clone_voice(self, name: str, reference_audio_paths: list[str]) -> str:
        import requests

        headers = {"xi-api-key": self.api_key}
        files = [("files", (Path(p).name, open(p, "rb"), "audio/wav")) for p in reference_audio_paths]
        data = {"name": name}

        response = requests.post(self.API_URL, headers=headers, data=data, files=files, timeout=120)
        response.raise_for_status()
        return response.json()["voice_id"]


class CoquiXTTSLocalVoiceCloneProvider(BaseVoiceCloneProvider):
    """Zero-shot voice cloning via Coqui XTTS-v2 -- free to run locally, no API
    key, but the model ships under the Coqui Public Model License 1.0.0,
    which is NON-COMMERCIAL by default (see 06_MODELS/tts/registry.json).
    Included for prototyping/reference only; do not ship in a commercial
    pipeline without separately clearing the license.

    XTTS has no server-side "register a voice" step: cloning just means
    keeping the reference wav(s) around and passing them to the TTS call.
    clone_voice() here validates the files exist and returns a "::"-joined
    path string that CoquiXTTSLocalTTSProvider.synthesize() knows how to
    unpack as its voice_id. This method does no torch/transformers import, so
    it is unaffected by the coqui-tts/transformers incompatibility noted on
    CoquiXTTSLocalTTSProvider below."""

    def clone_voice(self, name: str, reference_audio_paths: list[str]) -> str:
        if not reference_audio_paths:
            raise ValueError("reference_audio_paths must not be empty")
        for p in reference_audio_paths:
            if not Path(p).is_file():
                raise FileNotFoundError(f"Reference audio not found: {p}")
        return "::".join(str(Path(p).resolve()) for p in reference_audio_paths)


class VoiceCloner:
    def __init__(self, provider: BaseVoiceCloneProvider, asset_log_path: str = "./outputs/asset_records.jsonl"):
        self.provider = provider
        self.asset_log_path = asset_log_path

    def clone_voice(
        self,
        name: str,
        reference_audio_paths: list[str],
        character_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        license: Optional[str] = None,
    ) -> str:
        voice_id = self.provider.clone_voice(name=name, reference_audio_paths=reference_audio_paths)
        # cloned voices are remote provider resources, not local files -- record
        # a pseudo-URI so the asset ledger still has a stable identifier to key on.
        record = AssetRecord(
            asset_id=new_asset_id("voice"),
            file_path=f"remote://{type(self.provider).__name__}/voice/{voice_id}",
            character_id=character_id,
            episode_id=episode_id,
            model=type(self.provider).__name__,
            prompt=name,
            license=license,
            created_at=now_iso(),
        )
        write_asset_record(record, self.asset_log_path)
        return voice_id
