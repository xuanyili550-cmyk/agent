"""口型同步（lip-sync）生成：可插拔的 provider 抽象。

与 07_GENERATION 其他 provider 相同的模式：API 客户端占位实现，从环境变量读 key，
缺失则抛 NotConfiguredError。

流水线位置：视频（video_generator.py）和配音（tts_generator.py）都生成完之后，
把两者送给口型同步服务，让人物嘴型对上台词；产物再进入 08_QC 与 09_POST。
为什么做 Provider 抽象：口型同步服务商更迭很快，接口统一成 ``sync(video, audio) -> path``
后，编排层与素材登记逻辑（LipSyncGenerator）不用跟着改。
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


class BaseLipSyncProvider(ABC):
    """口型同步 provider 的抽象接口。"""

    @abstractmethod
    def sync(
        self,
        video_path: str,
        audio_path: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
    ) -> str:
        """返回口型同步后视频的本地文件路径。"""


class SyncSoLipSyncProvider(BaseLipSyncProvider):
    """sync.so 口型同步 API 客户端（骨架）：提交任务，按任务 id 约定产物路径。"""

    API_URL = "https://api.sync.so/v2/generate"

    def __init__(self):
        """从 SYNC_API_KEY 读取凭证；缺失立即报错。"""
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
        """POST 视频 + 音频 URL 建任务；请求经 request_with_retry 带退避重试。

        注意：这里只是骨架——没有轮询任务状态和下载产物，返回的是按任务 id 约定的输出路径。
        """
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "model": "lipsync-1.9.0-beta",
            "input": [
                {"type": "video", "url": video_path},
                {"type": "audio", "url": audio_path},
            ],
        }

        response = request_with_retry("POST", self.API_URL, json=payload, headers=headers)
        task = response.json()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # 响应里没有 id 时退回随机文件名，保证路径唯一
        out_path = out_dir / f"{task.get('id', uuid.uuid4().hex)}.mp4"
        return str(out_path)


class LipSyncGenerator:
    """口型同步门面：调用 provider 并把产物登记到素材台账。"""

    def __init__(self, provider: BaseLipSyncProvider, asset_log_path: str = "./outputs/asset_records.jsonl"):
        """注入 provider 与台账路径。"""
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
        """执行口型同步并写 AssetRecord；返回产物路径。

        ``prompt`` 字段记录输入的视频 / 音频路径，让台账能追溯这条产物由哪两份素材合成。
        """
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
