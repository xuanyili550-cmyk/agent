from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Literal, Optional


@dataclass
class AnalyticsItem:
    """统一的接入记录。kind="event" 是用户级事件（留存/CTR/收入都靠它），
    kind="metric" 是平台按天聚合数（YouTube/TikTok 不给用户级数据）。"""

    kind: Literal["event", "metric"]
    timestamp: datetime
    episode_id: Optional[str] = None
    # event 字段
    user_id: Optional[str] = None
    event_type: str = ""
    surface: Optional[str] = None
    amount_usd: Optional[float] = None
    experiment: Optional[str] = None
    variant: Optional[str] = None
    # metric 字段
    remote_id: Optional[str] = None
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    watch_time_minutes: float = 0.0
    estimated_revenue_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def dedupe_key(self) -> str:
        """同一条事件重复导入时 id 相同，主键冲突即去重。"""
        basis = f"{self.kind}|{self.user_id}|{self.episode_id}|{self.event_type}|{self.surface}|{self.timestamp.isoformat()}|{self.remote_id}"
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()


class AnalyticsSource(ABC):
    @abstractmethod
    def fetch(self, since: Optional[datetime] = None) -> Iterator[AnalyticsItem]:
        raise NotImplementedError
