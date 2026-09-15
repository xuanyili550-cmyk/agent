"""
剧集清单（Episode Manifest）：一集成片的元数据模型及其 JSON 读写。

在流水线中的位置：处于生产链路末端。上游（分镜生成、视频合成、QC）产出的成片、字幕、分镜 id、质检报告引用，
都汇总到一份 ``EpisodeManifest`` JSON 里；下游的多平台发布模块读取它，并把每个平台的发布状态回写到
``publish_status_per_platform``。

为什么把发布状态按平台分开存：同一集会分发到多个平台（抖音 / YouTube / TikTok ...），各平台审核进度、远端 id、
失败原因互不相关，按平台维度记录才能做逐平台重试与排查。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class PublishStatus(str, Enum):
    """单个平台上的发布状态机取值。继承 ``str`` 是为了能直接序列化成 JSON 字符串。"""

    NOT_SUBMITTED = "not_submitted"
    PENDING = "pending"
    IN_REVIEW = "in_review"
    PUBLISHED = "published"
    REJECTED = "rejected"
    FAILED = "failed"


class PlatformPublishStatus(BaseModel):
    """某一个平台的发布记录：状态、平台侧的远端 id / URL、最近一次尝试时间和错误信息。"""

    platform: str
    status: PublishStatus = PublishStatus.NOT_SUBMITTED
    remote_id: str | None = None
    remote_url: str | None = None
    last_attempt_at: datetime | None = None
    error_message: str | None = None


class EpisodeManifest(BaseModel):
    """一集成片的完整元数据。

    ``episode_number`` 从 1 开始、``duration_sec`` 必须为正，由 Field 约束在构造时校验；
    ``is_free`` / ``unlock_price_credits`` 支撑"前几集免费、后续付费解锁"的短剧商业模式。
    """

    episode_id: str
    series_id: str
    episode_number: int = Field(ge=1)
    title: str
    video_path: str
    duration_sec: float = Field(gt=0)
    subtitle_path: str | None = None
    shot_ids: list[str] = Field(default_factory=list)
    qc_report_ref: str | None = None
    publish_status_per_platform: dict[str, PlatformPublishStatus] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime | None = None
    language: str = "en"
    is_free: bool = True
    unlock_price_credits: int | None = None

    @field_validator("shot_ids")
    @classmethod
    def _non_empty_shot_ids(cls, v: list[str]) -> list[str]:
        """校验 ``shot_ids`` 非空：一集成片必须由至少一个分镜组成。"""
        # 剧集至少应引用一个分镜，否则说明上游流水线未产出内容
        if not v:
            raise ValueError("shot_ids must not be empty")
        return v

    @classmethod
    def load(cls, path: str | Path) -> EpisodeManifest:
        """从 JSON 文件读取并校验一份剧集清单。"""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        """把清单以缩进 JSON 写到文件；``exclude_none=False`` 保留 None 字段，让文件结构始终完整、便于 diff。"""
        Path(path).write_text(self.model_dump_json(indent=2, exclude_none=False), encoding="utf-8")


if __name__ == "__main__":
    import sys

    # 命令行用法：python episode_manifest.py a.json b.json ... 逐个校验清单文件
    for manifest_path in sys.argv[1:]:
        m = EpisodeManifest.load(manifest_path)
        print(f"OK: {manifest_path} -> {m.episode_id} ({m.duration_sec}s)")
