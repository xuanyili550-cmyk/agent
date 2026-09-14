from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import PayloadValidationError, PublishClient, PublishResult  # type: ignore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "10_EPISODES"))
from episode_manifest import EpisodeManifest, PublishStatus  # type: ignore  # noqa: E402

try:
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    GOOGLE_LIBS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when deps are missing
    GOOGLE_LIBS_AVAILABLE = False

YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
VALID_PRIVACY_STATUSES = {"public", "unlisted", "private"}


class YouTubePublishClient(PublishClient):
    """Uploads an EpisodeManifest as a YouTube Short via the YouTube Data API v3.

    Auth: OAuth2 "installed app" flow. Client secrets and the cached user token
    are read from local paths (never hardcoded) so no credentials live in source:
      - YOUTUBE_CLIENT_SECRETS_PATH: path to the OAuth client_secret.json from
        Google Cloud Console (Desktop app credential type).
      - YOUTUBE_TOKEN_PATH: path where the authorized user token is cached after
        the first interactive consent flow (refreshed automatically afterwards).
    """

    platform_name = "youtube"

    def __init__(
        self,
        client_secrets_path: str | None = None,
        token_path: str | None = None,
        category_id: str = "24",
        privacy_status: str = "public",
    ) -> None:
        self.client_secrets_path = client_secrets_path or os.environ.get("YOUTUBE_CLIENT_SECRETS_PATH", "")
        self.token_path = token_path or os.environ.get(
            "YOUTUBE_TOKEN_PATH",
            str(Path.home() / ".config" / "ai_short_drama" / "youtube_token.json"),
        )
        self.category_id = category_id
        self.privacy_status = privacy_status
        self._service: Any = None

    def build_payload(self, episode: EpisodeManifest) -> dict[str, Any]:
        return {
            "video_path": episode.video_path,
            "snippet": {
                "title": episode.title[:100],
                "description": (f"{episode.title}\n\nEpisode {episode.episode_number} of series {episode.series_id}.\n#shorts #shortdrama")[:5000],
                "tags": [episode.series_id, "shortdrama", "shorts"],
                "categoryId": self.category_id,
            },
            "status": {
                "privacyStatus": self.privacy_status,
                "selfDeclaredMadeForKids": False,
            },
        }

    def validate_payload(self, payload: dict[str, Any]) -> None:
        snippet = payload.get("snippet", {})
        title = snippet.get("title", "")
        if not title:
            raise PayloadValidationError("youtube payload missing snippet.title")
        if len(title) > 100:
            raise PayloadValidationError("youtube title exceeds 100 characters")
        status = payload.get("status", {})
        if status.get("privacyStatus") not in VALID_PRIVACY_STATUSES:
            raise PayloadValidationError(f"invalid privacyStatus, must be one of {VALID_PRIVACY_STATUSES}")
        if not payload.get("video_path"):
            raise PayloadValidationError("youtube payload missing video_path")

    def _get_credentials(self) -> Any:
        if not GOOGLE_LIBS_AVAILABLE:
            raise RuntimeError("google-api-python-client / google-auth-oauthlib not installed")
        creds = None
        token_file = Path(self.token_path)
        if token_file.exists():
            creds = Credentials.from_authorized_user_file(str(token_file), [YOUTUBE_UPLOAD_SCOPE])
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(GoogleAuthRequest())
            else:
                if not self.client_secrets_path or not Path(self.client_secrets_path).exists():
                    raise RuntimeError("YOUTUBE_CLIENT_SECRETS_PATH is not set or file does not exist; cannot run the OAuth consent flow")
                flow = InstalledAppFlow.from_client_secrets_file(self.client_secrets_path, [YOUTUBE_UPLOAD_SCOPE])
                creds = flow.run_local_server(port=0)
            token_file.parent.mkdir(parents=True, exist_ok=True)
            token_file.write_text(creds.to_json(), encoding="utf-8")
        return creds

    def _get_service(self) -> Any:
        if self._service is None:
            creds = self._get_credentials()
            self._service = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=creds)
        return self._service

    def fetch_status(self, remote_id: str) -> tuple[PublishStatus, str | None, str | None]:
        """videos.list(part=status,processingDetails)：uploadStatus=processed 且不是 rejected 才算发布完成。"""
        service = self._get_service()
        try:
            response = service.videos().list(part="status,processingDetails", id=remote_id).execute()
        except HttpError as exc:
            return PublishStatus.FAILED, None, str(exc)
        items = response.get("items", [])
        if not items:
            return PublishStatus.FAILED, None, f"video {remote_id} not found"
        status = items[0].get("status", {})
        upload_status = status.get("uploadStatus")
        url = f"https://www.youtube.com/watch?v={remote_id}"
        if upload_status == "rejected":
            return PublishStatus.REJECTED, url, status.get("rejectionReason")
        if upload_status == "failed":
            return PublishStatus.FAILED, url, status.get("failureReason")
        if upload_status == "processed":
            return PublishStatus.PUBLISHED, url, None
        return PublishStatus.IN_REVIEW, url, None  # uploaded / processing

    def _do_upload(self, episode: EpisodeManifest, payload: dict[str, Any]) -> PublishResult:
        service = self._get_service()
        body = {"snippet": payload["snippet"], "status": payload["status"]}
        media = MediaFileUpload(payload["video_path"], chunksize=-1, resumable=True, mimetype="video/mp4")
        request = service.videos().insert(part="snippet,status", body=body, media_body=media)
        try:
            response = None
            while response is None:
                _status, response = request.next_chunk()
            video_id = response["id"]
            # 上传完成 != 发布完成：YouTube 还要转码/审核，先记 IN_REVIEW，由 fetch_status 轮询确认
            return PublishResult(
                platform=self.platform_name,
                episode_id=episode.episode_id,
                status=PublishStatus.IN_REVIEW,
                remote_id=video_id,
                remote_url=f"https://www.youtube.com/watch?v={video_id}",
                payload=payload,
            )
        except HttpError as exc:
            return PublishResult(
                platform=self.platform_name,
                episode_id=episode.episode_id,
                status=PublishStatus.FAILED,
                error_message=str(exc),
                payload=payload,
            )
