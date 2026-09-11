from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import PayloadValidationError, PublishClient, PublishResult  # type: ignore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "10_EPISODES"))
from episode_manifest import EpisodeManifest, PublishStatus  # type: ignore  # noqa: E402

TIKTOK_API_BASE = "https://open.tiktokapis.com/v2"
INIT_ENDPOINT = f"{TIKTOK_API_BASE}/post/publish/video/init/"
STATUS_ENDPOINT = f"{TIKTOK_API_BASE}/post/publish/status/fetch/"
VALID_PRIVACY_LEVELS = {
    "PUBLIC_TO_EVERYONE",
    "MUTUAL_FOLLOW_FRIENDS",
    "SELF_ONLY",
}
UPLOAD_CHUNK_SIZE = 10 * 1024 * 1024


class TikTokPublishClient(PublishClient):
    """Uploads an EpisodeManifest via the TikTok Content Posting API (v2).

    Auth: a long-lived user access token obtained through TikTok's OAuth2 app
    flow (never hardcoded here). Read from TIKTOK_ACCESS_TOKEN env var.
    Flow implemented: FILE_UPLOAD source -> POST .../video/init/ to get a
    publish_id + upload_url, PUT the video bytes in chunks, then poll
    .../status/fetch/ for the publish result.
    """

    platform_name = "tiktok"

    def __init__(
        self,
        access_token: str | None = None,
        privacy_level: str = "SELF_ONLY",
        disable_duet: bool = False,
        disable_comment: bool = False,
        disable_stitch: bool = False,
        timeout_sec: int = 30,
    ) -> None:
        self.access_token = access_token or os.environ.get("TIKTOK_ACCESS_TOKEN", "")
        self.privacy_level = privacy_level
        self.disable_duet = disable_duet
        self.disable_comment = disable_comment
        self.disable_stitch = disable_stitch
        self.timeout_sec = timeout_sec

    def build_payload(self, episode: EpisodeManifest) -> dict[str, Any]:
        video_path = Path(episode.video_path)
        video_size = video_path.stat().st_size if video_path.exists() else 0
        return {
            "post_info": {
                "title": episode.title[:2200],
                "privacy_level": self.privacy_level,
                "disable_duet": self.disable_duet,
                "disable_comment": self.disable_comment,
                "disable_stitch": self.disable_stitch,
                "video_cover_timestamp_ms": 1000,
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": min(UPLOAD_CHUNK_SIZE, max(video_size, 1)),
                "total_chunk_count": max(
                    1, -(-video_size // UPLOAD_CHUNK_SIZE)
                )  # ceil division
                if video_size
                else 1,
            },
            "video_path": episode.video_path,
        }

    def validate_payload(self, payload: dict[str, Any]) -> None:
        post_info = payload.get("post_info", {})
        if not post_info.get("title"):
            raise PayloadValidationError("tiktok payload missing post_info.title")
        if post_info["privacy_level"] not in VALID_PRIVACY_LEVELS:
            raise PayloadValidationError(
                f"invalid privacy_level, must be one of {VALID_PRIVACY_LEVELS}"
            )
        source_info = payload.get("source_info", {})
        if source_info.get("source") != "FILE_UPLOAD":
            raise PayloadValidationError("only FILE_UPLOAD source is implemented")
        if not payload.get("video_path"):
            raise PayloadValidationError("tiktok payload missing video_path")

    def _auth_headers(self) -> dict[str, str]:
        if not self.access_token:
            raise RuntimeError("TIKTOK_ACCESS_TOKEN is not set")
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        }

    def _do_upload(
        self, episode: EpisodeManifest, payload: dict[str, Any]
    ) -> PublishResult:
        init_body = {
            "post_info": payload["post_info"],
            "source_info": payload["source_info"],
        }
        init_resp = requests.post(
            INIT_ENDPOINT,
            json=init_body,
            headers=self._auth_headers(),
            timeout=self.timeout_sec,
        )
        init_data = init_resp.json()
        error = init_data.get("error", {})
        if init_resp.status_code != 200 or error.get("code") not in (None, "ok"):
            return PublishResult(
                platform=self.platform_name,
                episode_id=episode.episode_id,
                status=PublishStatus.FAILED,
                error_message=error.get("message", f"HTTP {init_resp.status_code}"),
                payload=payload,
            )

        publish_id = init_data["data"]["publish_id"]
        upload_url = init_data["data"]["upload_url"]

        video_size = payload["source_info"]["video_size"]
        with open(payload["video_path"], "rb") as f:
            video_bytes = f.read()
        put_resp = requests.put(
            upload_url,
            data=video_bytes,
            headers={
                "Content-Type": "video/mp4",
                "Content-Range": f"bytes 0-{max(video_size - 1, 0)}/{video_size}",
            },
            timeout=self.timeout_sec,
        )
        if put_resp.status_code not in (200, 201):
            return PublishResult(
                platform=self.platform_name,
                episode_id=episode.episode_id,
                status=PublishStatus.FAILED,
                remote_id=publish_id,
                error_message=f"video upload failed: HTTP {put_resp.status_code}",
                payload=payload,
            )

        status_resp = requests.post(
            STATUS_ENDPOINT,
            json={"publish_id": publish_id},
            headers=self._auth_headers(),
            timeout=self.timeout_sec,
        )
        status_data = status_resp.json().get("data", {})
        publish_status = status_data.get("status", "PROCESSING_DOWNLOAD")

        if publish_status == "PUBLISH_COMPLETE":
            return PublishResult(
                platform=self.platform_name,
                episode_id=episode.episode_id,
                status=PublishStatus.PUBLISHED,
                remote_id=publish_id,
                remote_url=status_data.get("publicaly_available_post_id", [None])[0],
                payload=payload,
            )
        if publish_status == "FAILED":
            return PublishResult(
                platform=self.platform_name,
                episode_id=episode.episode_id,
                status=PublishStatus.FAILED,
                remote_id=publish_id,
                error_message=status_data.get("fail_reason", "unknown failure"),
                payload=payload,
            )
        return PublishResult(
            platform=self.platform_name,
            episode_id=episode.episode_id,
            status=PublishStatus.IN_REVIEW,
            remote_id=publish_id,
            payload=payload,
        )
