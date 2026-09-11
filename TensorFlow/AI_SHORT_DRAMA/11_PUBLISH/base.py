from __future__ import annotations

import abc
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "10_EPISODES"))
from episode_manifest import EpisodeManifest, PublishStatus  # type: ignore  # noqa: E402


class PayloadValidationError(ValueError):
    pass


@dataclass
class PublishResult:
    platform: str
    episode_id: str
    status: PublishStatus
    remote_id: str | None = None
    remote_url: str | None = None
    dry_run: bool = False
    payload: dict[str, Any] | None = None
    error_message: str | None = None
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PublishClient(abc.ABC):
    platform_name: str

    @abc.abstractmethod
    def build_payload(self, episode: EpisodeManifest) -> dict[str, Any]:
        """Turn an EpisodeManifest into the platform-specific request payload."""

    @abc.abstractmethod
    def validate_payload(self, payload: dict[str, Any]) -> None:
        """Raise PayloadValidationError if payload is malformed. Must not hit the network."""

    @abc.abstractmethod
    def _do_upload(
        self, episode: EpisodeManifest, payload: dict[str, Any]
    ) -> PublishResult:
        """Perform the real network call. Only invoked when dry_run=False."""

    def upload(
        self, episode: EpisodeManifest, dry_run: bool = False
    ) -> PublishResult:
        payload = self.build_payload(episode)
        self.validate_payload(payload)
        if dry_run:
            return PublishResult(
                platform=self.platform_name,
                episode_id=episode.episode_id,
                status=PublishStatus.PENDING,
                dry_run=True,
                payload=payload,
            )
        return self._do_upload(episode, payload)
