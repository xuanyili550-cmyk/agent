"""声音克隆：可插拔的 provider 抽象。

与 07_GENERATION 其他 provider 相同的模式：API 客户端从环境变量读 key，缺失则抛 NotConfiguredError。

流水线位置：角色定妆阶段——先用参考音频为每个角色克隆一个 voice_id，再交给 tts_generator.py
按台词合成配音，保证同一角色跨集音色一致。
为什么做 Provider 抽象：付费云端（ElevenLabs）和本地开源（Coqui XTTS）的"克隆"语义完全不同
（前者是服务端注册资源，后者只是保留参考 wav 路径），统一成 ``clone_voice() -> voice_id`` 后，
TTS 层不需要知道差别。
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
    """声音克隆 provider 的抽象接口。"""

    @abstractmethod
    def clone_voice(self, name: str, reference_audio_paths: list[str]) -> str:
        """返回 provider 侧的 voice_id，可直接传给对应的 TTS provider。"""


class ElevenLabsVoiceCloneProvider(BaseVoiceCloneProvider):
    """ElevenLabs 即时声音克隆：上传参考音频到 /voices/add，服务端返回 voice_id。"""

    API_URL = "https://api.elevenlabs.io/v1/voices/add"

    def __init__(self):
        """从 ELEVENLABS_API_KEY 读取凭证；缺失立即报错，避免到真正调用时才发现。"""
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise NotConfiguredError("ELEVENLABS_API_KEY is not set")

    def clone_voice(self, name: str, reference_audio_paths: list[str]) -> str:
        """以 multipart 表单上传所有参考 wav，返回服务端分配的 voice_id。"""
        import requests

        headers = {"xi-api-key": self.api_key}
        files = [("files", (Path(p).name, open(p, "rb"), "audio/wav")) for p in reference_audio_paths]
        data = {"name": name}

        response = requests.post(self.API_URL, headers=headers, data=data, files=files, timeout=120)
        response.raise_for_status()
        return response.json()["voice_id"]


class CoquiXTTSLocalVoiceCloneProvider(BaseVoiceCloneProvider):
    """基于 Coqui XTTS-v2 的零样本声音克隆——本地免费运行、无需 API key，但模型采用
    Coqui Public Model License 1.0.0，默认**非商用**（见 06_MODELS/tts/registry.json）。
    仅作原型 / 参考用途；未单独取得许可前不要用于商业流水线。

    XTTS 没有服务端"注册声音"这一步：克隆就是把参考 wav 留着、在 TTS 调用时传进去。
    这里的 clone_voice() 只校验文件存在，返回用 "::" 拼接的路径字符串，
    CoquiXTTSLocalTTSProvider.synthesize() 知道如何把它当 voice_id 拆开。
    本方法不 import torch/transformers，因此不受 CoquiXTTSLocalTTSProvider 处
    记录的 coqui-tts/transformers 版本不兼容问题影响。"""

    def clone_voice(self, name: str, reference_audio_paths: list[str]) -> str:
        """校验参考音频非空且都存在，返回绝对路径用 "::" 拼接的伪 voice_id。"""
        if not reference_audio_paths:
            raise ValueError("reference_audio_paths must not be empty")
        for p in reference_audio_paths:
            if not Path(p).is_file():
                raise FileNotFoundError(f"Reference audio not found: {p}")
        # 用 resolve() 转绝对路径：TTS 可能在别的工作目录被调用
        return "::".join(str(Path(p).resolve()) for p in reference_audio_paths)


class VoiceCloner:
    """声音克隆门面：调用 provider 克隆，并把结果登记到素材台账。"""

    def __init__(self, provider: BaseVoiceCloneProvider, asset_log_path: str = "./outputs/asset_records.jsonl"):
        """注入 provider 与台账路径。"""
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
        """克隆声音并写一条 AssetRecord；返回 voice_id。"""
        voice_id = self.provider.clone_voice(name=name, reference_audio_paths=reference_audio_paths)
        # 克隆出来的声音是 provider 侧的远程资源而不是本地文件——记一个伪 URI，
        # 让素材台账仍然有一个稳定的标识可以关联。
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
