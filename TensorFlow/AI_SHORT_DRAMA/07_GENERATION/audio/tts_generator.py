"""
Text-to-speech generation via a pluggable provider abstraction. Same pattern
as video_generator.py: concrete providers are API client placeholders that
read their key from an env var and raise NotConfiguredError if missing.
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
from http_retry import request_with_retry  # noqa: E402


class BaseTTSProvider(ABC):
    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
    ) -> str:
        """Returns a local file path to the generated audio."""


class ElevenLabsTTSProvider(BaseTTSProvider):
    API_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    def __init__(self):
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise NotConfiguredError("ELEVENLABS_API_KEY is not set")

    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
    ) -> str:
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {"text": text, "model_id": "eleven_multilingual_v2"}

        response = request_with_retry("POST", self.API_URL_TEMPLATE.format(voice_id=voice_id), json=payload, headers=headers)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{uuid.uuid4().hex}.mp3"
        out_path.write_bytes(response.content)
        return str(out_path)


class BarkLocalTTSProvider(BaseTTSProvider):
    """Runs Suno Bark (MIT, free incl. commercial use) locally via transformers
    -- no API key. voice_id maps to a Bark speaker preset, e.g. "v2/en_speaker_6"
    or "v2/zh_speaker_4" (see https://huggingface.co/suno/bark for the full list)."""

    def __init__(self, model_id: str = "suno/bark-small", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        from transformers import pipeline

        self._pipe = pipeline("text-to-speech", model=self.model_id, device=self.device)
        return self._pipe

    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
    ) -> str:
        import scipy.io.wavfile

        pipe = self._load()
        preprocess_params = {"voice_preset": voice_id} if voice_id else {}
        output = pipe(text, preprocess_params=preprocess_params)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{uuid.uuid4().hex}.wav"
        scipy.io.wavfile.write(str(out_path), rate=output["sampling_rate"], data=output["audio"].squeeze())
        return str(out_path)


class CoquiXTTSLocalTTSProvider(BaseTTSProvider):
    """Speaks text in a cloned voice via Coqui XTTS-v2, using the "::"-joined
    reference-wav paths produced by CoquiXTTSLocalVoiceCloneProvider.clone_voice()
    as voice_id. Free to run locally, no API key -- but see the license
    caveat on CoquiXTTSLocalVoiceCloneProvider: non-commercial by default.

    BROKEN against this project's main environment (`transformers` 5.5.0):
    `pip install coqui-tts` (0.27.5) fails at import time --
    `ImportError: cannot import name 'isin_mps_friendly' from
    'transformers.pytorch_utils'` -- because coqui-tts's XTTS code imports a
    helper transformers 5.5.0 no longer exports. VERIFIED WORKING end-to-end
    (real 5.2s wav produced, cross-checked with ffprobe) from a separate venv:

        python3 -m venv .venv_xtts && source .venv_xtts/bin/activate
        pip install "transformers<5.0" coqui-tts pydantic
        pip install torch torchaudio
        pip install "coqui-tts[codec]"   # pulls in torchcodec

    Two more real, verified gotchas on macOS in that venv:
    1. torchcodec's bundled native libs fail to dlopen against this machine's
       Homebrew ffmpeg (`OSError: ... Library not loaded: @rpath/libavutil.
       57.dylib ... no LC_RPATH's found`) unless you point the loader at it:
       `export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix ffmpeg)/lib"`.
    2. First XTTS-v2 download blocks on an interactive TOS prompt (it ships
       under the non-commercial CPML, see the license caveat above) --
       `COQUI_TOS_AGREED=1` skips the prompt, but only set it once you've
       actually read and agreed to https://coqui.ai/cpml yourself; this repo
       does not set it for you.

    Not patched in this codebase (upstream version-compatibility gap between
    two third-party libraries, not a bug in this class); the main environment
    stays on transformers 5.5.0 for LocalTransformersProvider/BarkLocalTTSProvider/
    WhisperLocalASRProvider, same pattern as the Python 3.14/`datasets`/`dill`
    workaround documented in the project README."""

    def __init__(self, model_id: str = "tts_models/multilingual/multi-dataset/xtts_v2", language: str = "zh", device: str = "cuda"):
        self.model_id = model_id
        self.language = language
        self.device = device
        self._tts = None

    def _load(self):
        if self._tts is not None:
            return self._tts
        from TTS.api import TTS

        self._tts = TTS(self.model_id).to(self.device)
        return self._tts

    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
    ) -> str:
        tts = self._load()
        speaker_wavs = voice_id.split("::")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{uuid.uuid4().hex}.wav"
        tts.tts_to_file(text=text, speaker_wav=speaker_wavs, language=self.language, file_path=str(out_path))
        return str(out_path)


class TTSGenerator:
    def __init__(self, provider: BaseTTSProvider, asset_log_path: str = "./outputs/asset_records.jsonl"):
        self.provider = provider
        self.asset_log_path = asset_log_path

    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
        character_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        shot_id: Optional[str] = None,
        license: Optional[str] = None,
    ) -> str:
        file_path = self.provider.synthesize(text=text, voice_id=voice_id, output_dir=output_dir, seed=seed)
        record = AssetRecord(
            asset_id=new_asset_id("tts"),
            file_path=file_path,
            character_id=character_id,
            episode_id=episode_id,
            shot_id=shot_id,
            model=type(self.provider).__name__,
            prompt=text,
            seed=seed,
            license=license,
            created_at=now_iso(),
        )
        write_asset_record(record, self.asset_log_path)
        return file_path
