"""YouTube Analytics API v2（youtubeAnalytics.reports.query）。

在流水线中的位置：12_ANALYTICS/ingest 的平台指标源之一。11_PUBLISH/youtube 把剧集发上去之后，本模块把平台侧的播放 / 互动 / 收益
数据按天拉回来，产出 ``kind="metric"`` 的 ``AnalyticsItem``（YouTube 不提供用户级事件，所以只能是聚合指标）。

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

# 收益指标（estimatedRevenue）需要额外的 monetary 只读 scope
SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly", "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
METRICS = "views,likes,comments,shares,estimatedMinutesWatched,estimatedRevenue"


class YouTubeAnalyticsSource(AnalyticsSource):
    """从 YouTube Analytics 拉取本频道各视频按天的聚合指标。"""

    def __init__(self, video_episode_map: Optional[dict[str, str]] = None, token_path: Optional[str] = None, lookback_days: int = 28, service: Any = None):
        """初始化配置。

        - ``video_episode_map``：平台 video_id -> 我们的 episode_id；
        - ``token_path``：OAuth token 文件，缺省取环境变量 YOUTUBE_TOKEN_PATH，再缺省用 ~/.config/ai_short_drama/youtube_token.json；
        - ``lookback_days``：未指定 ``since`` 时默认回看的天数（28 天是 YouTube 后台的默认窗口）；
        - ``service``：可注入的 API client，测试时传入假对象即可完全离线。
        """
        self.video_episode_map = video_episode_map or {}
        self.token_path = token_path or os.environ.get("YOUTUBE_TOKEN_PATH", str(Path.home() / ".config" / "ai_short_drama" / "youtube_token.json"))
        self.lookback_days = lookback_days
        self._service = service  # 测试可注入假 service

    def _get_service(self):
        """懒加载 YouTube Analytics API client；google 相关库在函数内导入，不用 YouTube 时无需安装这些依赖。"""
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
        """查询 [since 或 今天 - lookback_days, 今天] 区间内按 day,video 维度的报表，每行产出一个 metric 记录。

        API 返回的是 columnHeaders + rows 的表格结构，先把表头名与每行 zip 成 dict 再取字段，避免依赖列顺序。
        """
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
            # 报表里的 day 是 "YYYY-MM-DD" 无时区，统一标成 UTC 零点
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
                # 未开通变现的频道 estimatedRevenue 可能是 None，多一层 `or 0.0` 兜底
                estimated_revenue_usd=float(rec.get("estimatedRevenue", 0.0) or 0.0),
                raw=rec,
            )
