"""
数据接入层的基础类型：统一记录 ``AnalyticsItem`` 与数据源抽象基类 ``AnalyticsSource``。

在流水线中的位置：12_ANALYTICS/ingest 的最底层。文件事件源、YouTube、TikTok 三种来源的数据形态差异很大
（用户级事件 vs 平台按天聚合指标），先在这里收敛成同一个 dataclass，下游的分析模块与入库逻辑才不必关心来源。

为什么把 event 与 metric 两类字段放在同一个 dataclass：两类记录要写进同一张表、走同一条去重逻辑，用 ``kind`` 区分
比维护两套类型 + 两套入库代码简单得多；不适用的字段保持默认值即可。
"""

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
    # 原始记录原样保留，方便排查与后续补字段时重放
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def dedupe_key(self) -> str:
        """同一条事件重复导入时 id 相同，主键冲突即去重。

        去重键由 kind + 用户 + 剧集 + 事件类型 + 展示位 + 时间戳 + 平台 id 拼接后取 sha1：
        既覆盖 event（靠 user/event_type/surface 区分），也覆盖 metric（靠 remote_id/timestamp 区分）。
        """
        basis = f"{self.kind}|{self.user_id}|{self.episode_id}|{self.event_type}|{self.surface}|{self.timestamp.isoformat()}|{self.remote_id}"
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()


class AnalyticsSource(ABC):
    """数据源抽象基类：任何来源只需实现 ``fetch``，产出 ``AnalyticsItem`` 迭代器。"""

    @abstractmethod
    def fetch(self, since: Optional[datetime] = None) -> Iterator[AnalyticsItem]:
        """拉取数据；``since`` 用于增量拉取（只返回该时间之后的记录），来源不支持时可忽略。"""
        raise NotImplementedError
