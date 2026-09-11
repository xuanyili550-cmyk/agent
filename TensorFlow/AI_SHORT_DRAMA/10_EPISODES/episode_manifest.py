from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class PublishStatus(str, Enum):
    NOT_SUBMITTED = "not_submitted"
    PENDING = "pending"
    IN_REVIEW = "in_review"
    PUBLISHED = "published"
    REJECTED = "rejected"
    FAILED = "failed"


class PlatformPublishStatus(BaseModel):
    platform: str
    status: PublishStatus = PublishStatus.NOT_SUBMITTED
    remote_id: str | None = None
    remote_url: str | None = None
    last_attempt_at: datetime | None = None
    error_message: str | None = None


class EpisodeManifest(BaseModel):
    episode_id: str
    series_id: str
    episode_number: int = Field(ge=1)
    title: str
    video_path: str
    duration_sec: float = Field(gt=0)
    subtitle_path: str | None = None
    shot_ids: list[str] = Field(default_factory=list)
    qc_report_ref: str | None = None
    publish_status_per_platform: dict[str, PlatformPublishStatus] = Field(
        default_factory=dict
    )
    created_at: datetime
    updated_at: datetime | None = None
    language: str = "en"
    is_free: bool = True
    unlock_price_credits: int | None = None

    @field_validator("shot_ids")
    @classmethod
    def _non_empty_shot_ids(cls, v: list[str]) -> list[str]:
        # 剧集至少应引用一个分镜，否则说明上游流水线未产出内容
        if not v:
            raise ValueError("shot_ids must not be empty")
        return v

    @classmethod
    def load(cls, path: str | Path) -> "EpisodeManifest":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            self.model_dump_json(indent=2, exclude_none=False), encoding="utf-8"
        )


if __name__ == "__main__":
    import sys

    for manifest_path in sys.argv[1:]:
        m = EpisodeManifest.load(manifest_path)
        print(f"OK: {manifest_path} -> {m.episode_id} ({m.duration_sec}s)")
