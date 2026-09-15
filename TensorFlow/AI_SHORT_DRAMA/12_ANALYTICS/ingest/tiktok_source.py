"""TikTok Display API：POST /v2/video/query/ 按 video_id 拿累计指标（view_count / like_count / ...）。

在流水线中的位置：12_ANALYTICS/ingest 的平台指标源之一，与 11_PUBLISH/tiktok 配套：发布记录里的 remote_id 就是这里要查询的 video_id。

TikTok 不提供按天的历史序列，只有当前累计值，所以每次拉取记成"今天"这一天的快照；
按天做差就是当天增量。access_token 与 11_PUBLISH/tiktok 共用（TIKTOK_ACCESS_TOKEN）。
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any, Callable, Iterator, Optional

from .base import AnalyticsItem, AnalyticsSource

VIDEO_QUERY_URL = "https://open.tiktokapis.com/v2/video/query/"
# 要返回的字段通过 query string 的 fields 参数指定
FIELDS = "id,view_count,like_count,comment_count,share_count,create_time,title"


class TikTokAnalyticsSource(AnalyticsSource):
    """从 TikTok Display API 拉取已发布视频的累计指标快照。"""

    def __init__(self, video_episode_map: Optional[dict[str, str]] = None, access_token: Optional[str] = None, http_post: Optional[Callable[..., Any]] = None):
        """初始化配置：``video_episode_map`` 的键就是要查询的 video_id 列表；``access_token`` 缺省读环境变量；``http_post`` 供测试注入假 HTTP 客户端。"""
        self.video_episode_map = video_episode_map or {}
        self.access_token = access_token or os.environ.get("TIKTOK_ACCESS_TOKEN", "")
        self._post = http_post  # 测试注入

    def _post_json(self, url: str, **kwargs):
        """发 POST 请求：优先用注入的客户端；否则在函数内导入 requests（不用 TikTok 时无需安装）并带 30 秒超时。"""
        if self._post is not None:
            return self._post(url, **kwargs)
        import requests

        return requests.post(url, timeout=30, **kwargs)

    def fetch(self, since: Optional[datetime] = None) -> Iterator[AnalyticsItem]:
        """按 20 个一批查询全部 video_id，把每个视频的累计指标记成"今天 UTC 零点"的一条 metric 记录。

        ``since`` 参数在这里没有实际作用：TikTok 只返回当前累计值，无法按时间过滤；保留签名是为了满足 ``AnalyticsSource`` 接口。
        """
        if not self.access_token:
            raise RuntimeError("TIKTOK_ACCESS_TOKEN 未设置")
        video_ids = list(self.video_episode_map) or []
        if not video_ids:
            return
        # 快照统一打成当天零点，便于与 YouTube 的按天指标对齐、按天做差
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(0, len(video_ids), 20):  # API 单次最多 20 个
            batch = video_ids[i : i + 20]
            resp = self._post_json(
                f"{VIDEO_QUERY_URL}?fields={FIELDS}",
                json={"filters": {"video_ids": batch}},
                headers={"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"},
            )
            data = resp.json()
            # TikTok 的错误不走 HTTP 状态码，成功时 error.code == "ok"，其它值都是失败
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
