"""YouTube Analytics API v2（youtubeAnalytics.reports.query）。

按天 + 按视频拉 views / likes / comments / shares / estimatedMinutesWatched / estimatedRevenue。
需要 OAuth 凭证（scope yt-analytics.readonly 与 yt-analytics-monetary.readonly），与
11_PUBLISH/youtube 共用 token 文件的方式一致。``video_episode_map`` 把平台 video_id 映射回我们的
episode_id（来自 publish_records.remote_id）。
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional

from .base import AnalyticsItem, AnalyticsSource

SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly", "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
METRICS = "views,likes,comments,shares,estimatedMinutesWatched,estimatedRevenue"


class YouTubeAnalyticsSource(AnalyticsSource):
    def __init__(self, video_episode_map: Optional[dict[str, str]] = None, token_path: Optional[str] = None, lookback_days: int = 28, service: Any = None):
        self.video_episode_map = video_episode_map or {}
        self.token_path = token_path or os.environ.get("YOUTUBE_TOKEN_PATH", str(Path.home() / ".config" / "ai_short_drama" / "youtube_token.json"))
        self.lookback_days = lookback_days
        self._service = service  # 测试可注入假 service

    def _get_service(self):
        if self._service is not None:
            return self._service
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        if not Path(self.token_path).exists():
            raise RuntimeError(f"YouTube token 不存在：{self.token_path}，先完成 11_PUBLISH/youtube 的 OAuth 授权")
        creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
        self._service = build("youtubeAnalytics", "v2", credentials=creds)
        return self._service

    def fetch(self, since: Optional[datetime] = None) -> Iterator[AnalyticsItem]:
        service = self._get_service()
        end = datetime.now(UTC).date()
        start = since.date() if since else end - timedelta(days=self.lookback_days)
        response = (
            service.reports()
            .query(ids="channel==MINE", startDate=start.isoformat(), endDate=end.isoformat(), metrics=METRICS, dimensions="day,video", sort="day")
            .execute()
        )
        headers = [h["name"] for h in response.get("columnHeaders", [])]
        for row in response.get("rows", []):
            rec = dict(zip(headers, row))
            day = datetime.fromisoformat(rec["day"]).replace(tzinfo=UTC)
            video_id = rec.get("video")
            yield AnalyticsItem(
                kind="metric",
                timestamp=day,
                remote_id=video_id,
                episode_id=self.video_episode_map.get(video_id),
                views=int(rec.get("views", 0)),
                likes=int(rec.get("likes", 0)),
                comments=int(rec.get("comments", 0)),
                shares=int(rec.get("shares", 0)),
                watch_time_minutes=float(rec.get("estimatedMinutesWatched", 0.0)),
                estimated_revenue_usd=float(rec.get("estimatedRevenue", 0.0) or 0.0),
                raw=rec,
            )
