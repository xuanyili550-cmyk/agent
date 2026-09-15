"""TikTok 发布客户端：通过 TikTok Content Posting API v2 的 FILE_UPLOAD 方式上传一集成片。

三步流程：POST /video/init/ 拿 publish_id + upload_url -> PUT 视频字节 -> POST /status/fetch/ 查结果。
默认 privacy_level 是 SELF_ONLY（仅自己可见）：TikTok 对未审核通过的第三方应用强制只能发私密内容，
而且首次接入时先私发再人工检查更安全，真正公开发布时再显式传 PUBLIC_TO_EVERYONE。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import requests

# parents[1] 是 11_PUBLISH（找 base），parents[2] 是项目根（找 10_EPISODES）
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
UPLOAD_CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB：TikTok 要求分块在 5MB~64MB 之间，取一个保守值


class TikTokPublishClient(PublishClient):
    """通过 TikTok Content Posting API（v2）上传 EpisodeManifest。

    鉴权：通过 TikTok OAuth2 应用流程获得的长期用户 access token（这里绝不硬编码），从 TIKTOK_ACCESS_TOKEN 环境变量读取。
    已实现的流程：FILE_UPLOAD 来源 -> POST .../video/init/ 拿到 publish_id + upload_url，
    分块 PUT 视频字节，然后轮询 .../status/fetch/ 得到发布结果。
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
        """参数优先，其次读 TIKTOK_ACCESS_TOKEN；token 为空不在这里报错（dry-run 不需要），等真正发请求时再查。"""
        self.access_token = access_token or os.environ.get("TIKTOK_ACCESS_TOKEN", "")
        self.privacy_level = privacy_level
        self.disable_duet = disable_duet
        self.disable_comment = disable_comment
        self.disable_stitch = disable_stitch
        self.timeout_sec = timeout_sec

    def build_payload(self, episode: EpisodeManifest) -> dict[str, Any]:
        """组装 /video/init/ 的 post_info + source_info，外加本地 video_path。

        文件不存在时 video_size 记 0 而不是抛错，让 dry-run 在只有 manifest、没有成片的环境下也能跑通校验。
        chunk_size / total_chunk_count 是 TikTok init 接口的必填字段，即使实际 PUT 时一次传完也要给出一致的值。
        """
        video_path = Path(episode.video_path)
        video_size = video_path.stat().st_size if video_path.exists() else 0
        return {
            "post_info": {
                "title": episode.title[:2200],  # TikTok 标题/描述上限 2200 字符
                "privacy_level": self.privacy_level,
                "disable_duet": self.disable_duet,
                "disable_comment": self.disable_comment,
                "disable_stitch": self.disable_stitch,
                "video_cover_timestamp_ms": 1000,  # 封面取第 1 秒的帧，避开开头可能的黑场
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": min(UPLOAD_CHUNK_SIZE, max(video_size, 1)),
                "total_chunk_count": max(1, -(-video_size // UPLOAD_CHUNK_SIZE))  # 向上取整
                if video_size
                else 1,
            },
            "video_path": episode.video_path,
        }

    def validate_payload(self, payload: dict[str, Any]) -> None:
        """本地校验：标题非空、privacy_level 合法、source 只能是 FILE_UPLOAD（PULL_FROM_URL 未实现）、video_path 存在。"""
        post_info = payload.get("post_info", {})
        if not post_info.get("title"):
            raise PayloadValidationError("tiktok payload missing post_info.title")
        if post_info["privacy_level"] not in VALID_PRIVACY_LEVELS:
            raise PayloadValidationError(f"invalid privacy_level, must be one of {VALID_PRIVACY_LEVELS}")
        source_info = payload.get("source_info", {})
        if source_info.get("source") != "FILE_UPLOAD":
            raise PayloadValidationError("only FILE_UPLOAD source is implemented")
        if not payload.get("video_path"):
            raise PayloadValidationError("tiktok payload missing video_path")

    def _auth_headers(self) -> dict[str, str]:
        """构造 Bearer 鉴权头；token 缺失在这里才报错，保证 dry-run 路径不会碰到它。"""
        if not self.access_token:
            raise RuntimeError("TIKTOK_ACCESS_TOKEN is not set")
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        }

    def fetch_status(self, remote_id: str) -> tuple[PublishStatus, str | None, str | None]:
        """POST /post/publish/status/fetch/：PUBLISH_COMPLETE / FAILED / 其余仍在处理。"""
        resp = requests.post(STATUS_ENDPOINT, json={"publish_id": remote_id}, headers=self._auth_headers(), timeout=self.timeout_sec)
        data = resp.json().get("data", {})
        status = data.get("status", "PROCESSING_DOWNLOAD")
        if status == "PUBLISH_COMPLETE":
            post_ids = data.get("publicaly_available_post_id") or []  # 字段名的拼写错误（publicaly）是 TikTok API 原样如此
            url = f"https://www.tiktok.com/@/video/{post_ids[0]}" if post_ids else None
            return PublishStatus.PUBLISHED, url, None
        if status == "FAILED":
            return PublishStatus.FAILED, None, data.get("fail_reason", "unknown failure")
        return PublishStatus.IN_REVIEW, None, None

    def _do_upload(self, episode: EpisodeManifest, payload: dict[str, Any]) -> PublishResult:
        """init -> PUT 视频 -> 查一次状态。每一步失败都转成 FAILED 结果返回（带上已拿到的 publish_id 便于追查），不抛异常。"""
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
        # TikTok 即使 HTTP 200 也可能在 body 的 error.code 里报错，"ok" 才是成功
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

        # 短剧单集通常几十 MB，直接整体读进内存一次 PUT，Content-Range 覆盖全部字节即可
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

        # 上传完立刻查一次状态：大多数情况下还在处理中，会返回 IN_REVIEW，交给 fetch_status 后续轮询
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
