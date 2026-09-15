"""YouTube 发布客户端：通过 YouTube Data API v3 把一集成片作为 Short 上传，并支持轮询转码/审核状态。

Google 相关库是可选依赖：没装时模块仍可导入，build_payload / validate_payload（dry-run 路径）照常工作，
只有真正上传或查状态时才会因 GOOGLE_LIBS_AVAILABLE=False 报错。这样 CI 和 publish_cli --dry-run 不需要装一整套 google 包。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# parents[1] 是 11_PUBLISH（找 base），parents[2] 是项目根（找 10_EPISODES）
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

YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"  # 只申请上传权限，最小授权范围
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
VALID_PRIVACY_STATUSES = {"public", "unlisted", "private"}


class YouTubePublishClient(PublishClient):
    """通过 YouTube Data API v3 把 EpisodeManifest 作为 YouTube Short 上传。

    鉴权：OAuth2 "installed app" 流程。客户端密钥和缓存的用户 token 都从本地路径读取（绝不硬编码），
    源码里不会出现任何凭据：
      - YOUTUBE_CLIENT_SECRETS_PATH：Google Cloud Console 下载的 OAuth client_secret.json（Desktop app 类型）的路径。
      - YOUTUBE_TOKEN_PATH：首次交互式授权后缓存已授权用户 token 的路径（之后自动刷新）。
    """

    platform_name = "youtube"

    def __init__(
        self,
        client_secrets_path: str | None = None,
        token_path: str | None = None,
        category_id: str = "24",
        privacy_status: str = "public",
    ) -> None:
        """参数优先，其次环境变量；token 默认缓存在 ~/.config/ai_short_drama/。category_id 24 是 YouTube 的 Entertainment 分类。"""
        self.client_secrets_path = client_secrets_path or os.environ.get("YOUTUBE_CLIENT_SECRETS_PATH", "")
        self.token_path = token_path or os.environ.get(
            "YOUTUBE_TOKEN_PATH",
            str(Path.home() / ".config" / "ai_short_drama" / "youtube_token.json"),
        )
        self.category_id = category_id
        self.privacy_status = privacy_status
        self._service: Any = None  # 懒加载的 API client，第一次真正上传/查状态时才走 OAuth

    def build_payload(self, episode: EpisodeManifest) -> dict[str, Any]:
        """组装 videos.insert 需要的 snippet/status，外加本地 video_path（上传时用，不会发给 API）。

        标题截到 100、描述截到 5000 是 YouTube 的硬限制；#shorts 标签让平台把它识别为 Short。
        """
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
                "selfDeclaredMadeForKids": False,  # COPPA 声明字段，短剧不是儿童内容，必须显式给 False
            },
        }

    def validate_payload(self, payload: dict[str, Any]) -> None:
        """本地校验：标题非空且 <= 100 字符、privacyStatus 合法、video_path 存在。不访问网络。"""
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
        """取得可用的 OAuth 凭据：优先读缓存 token；过期且有 refresh_token 就刷新；否则跑一次本地浏览器授权流程并缓存。"""
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
                creds = flow.run_local_server(port=0)  # port=0 让系统随机挑空闲端口，避免和本机其他服务冲突
            token_file.parent.mkdir(parents=True, exist_ok=True)
            token_file.write_text(creds.to_json(), encoding="utf-8")  # 刷新/新授权后都回写缓存，下次不用再交互
        return creds

    def _get_service(self) -> Any:
        """懒构造并缓存 YouTube API service 对象。"""
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
        return PublishStatus.IN_REVIEW, url, None  # uploaded / processing：已上传但还在转码，继续轮询

    def _do_upload(self, episode: EpisodeManifest, payload: dict[str, Any]) -> PublishResult:
        """用 resumable 分块上传把视频推到 YouTube；API 错误不抛出，转成 FAILED 结果返回，让上游统一记录。"""
        service = self._get_service()
        body = {"snippet": payload["snippet"], "status": payload["status"]}
        # chunksize=-1 表示一次性上传整个文件（resumable 只用于断线续传），短剧单集体积小，不需要分块
        media = MediaFileUpload(payload["video_path"], chunksize=-1, resumable=True, mimetype="video/mp4")
        request = service.videos().insert(part="snippet,status", body=body, media_body=media)
        try:
            response = None
            while response is None:  # next_chunk 在上传未完成时返回 (status, None)，完成时返回 (None, response)
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
