"""TikTok Display API：POST /v2/video/query/ 按 video_id 拿累计指标（view_count / like_count / ...）。

TikTok 不提供按天的历史序列，只有当前累计值，所以每次拉取记成"今天"这一天的快照；
按天做差就是当天增量。access_token 与 11_PUBLISH/tiktok 共用（TIKTOK_ACCESS_TOKEN）。
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any, Callable, Iterator, Optional

from .base import AnalyticsItem, AnalyticsSource

VIDEO_QUERY_URL = "https://open.tiktokapis.com/v2/video/query/"
FIELDS = "id,view_count,like_count,comment_count,share_count,create_time,title"


class TikTokAnalyticsSource(AnalyticsSource):
    def __init__(self, video_episode_map: Optional[dict[str, str]] = None, access_token: Optional[str] = None, http_post: Optional[Callable[..., Any]] = None):
        self.video_episode_map = video_episode_map or {}
        self.access_token = access_token or os.environ.get("TIKTOK_ACCESS_TOKEN", "")
        self._post = http_post  # 测试注入

    def _post_json(self, url: str, **kwargs):
        if self._post is not None:
            return self._post(url, **kwargs)
        import requests

        return requests.post(url, timeout=30, **kwargs)

    def fetch(self, since: Optional[datetime] = None) -> Iterator[AnalyticsItem]:
        if not self.access_token:
            raise RuntimeError("TIKTOK_ACCESS_TOKEN 未设置")
        video_ids = list(self.video_episode_map) or []
        if not video_ids:
            return
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(0, len(video_ids), 20):  # API 单次最多 20 个
            batch = video_ids[i : i + 20]
            resp = self._post_json(
                f"{VIDEO_QUERY_URL}?fields={FIELDS}",
                json={"filters": {"video_ids": batch}},
                headers={"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"},
            )
            data = resp.json()
            error = data.get("error", {})
            if error.get("code") not in (None, "ok"):
                raise RuntimeError(f"TikTok video/query 失败：{error}")
            for video in data.get("data", {}).get("videos", []):
                yield AnalyticsItem(
                    kind="metric",
                    timestamp=today,
                    remote_id=video["id"],
                    episode_id=self.video_episode_map.get(video["id"]),
                    views=int(video.get("view_count", 0)),
                    likes=int(video.get("like_count", 0)),
                    comments=int(video.get("comment_count", 0)),
                    shares=int(video.get("share_count", 0)),
                    raw=video,
                )
