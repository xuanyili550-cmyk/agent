"""
文件事件源：把自家 App / SDK 导出的 JSONL 用户级事件读成统一的 ``AnalyticsItem``。

在流水线中的位置：12_ANALYTICS/ingest 的三种数据源之一（另两种是 YouTube / TikTok 平台指标）。
留存、CTR、收入、A/B 实验都依赖用户级事件，而平台 API 只给聚合指标，所以这是最核心的一路数据。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Optional

from .base import AnalyticsItem, AnalyticsSource


class FileEventSource(AnalyticsSource):
    """jsonl 用户级事件：每行 {user_id, episode_id, event_type, timestamp, surface?, amount_usd?, experiment?, variant?}。
    自家 App 的埋点导出、或从数据仓库拉下来的文件都走这里。"""

    def __init__(self, path: str | Path):
        """记录事件文件路径；不在构造时读文件，真正读取推迟到 ``fetch``。"""
        self.path = Path(path)

    def fetch(self, since: Optional[datetime] = None) -> Iterator[AnalyticsItem]:
        """逐行读取事件并产出 ``kind="event"`` 的 ``AnalyticsItem``；给定 ``since`` 时跳过更早的事件（增量拉取）。

        时间戳统一规范为带时区的 UTC：ISO 字符串里的 "Z" 后缀 ``fromisoformat`` 在部分 Python 版本不认，先替换成 "+00:00"；
        完全没有时区信息的时间戳按 UTC 处理，避免后续与 ``since`` 比较时出现 naive/aware 混用报错。
        """
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ts = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if since and ts < since:
                continue
            yield AnalyticsItem(
                kind="event",
                timestamp=ts,
                user_id=row.get("user_id"),
                episode_id=row.get("episode_id"),
                event_type=row.get("event_type", ""),
                surface=row.get("surface"),
                # amount_usd 只有购买类事件才有；显式判 None 而不是用 truthiness，避免 0 元订单被丢掉
                amount_usd=float(row["amount_usd"]) if row.get("amount_usd") is not None else None,
                experiment=row.get("experiment"),
                variant=row.get("variant"),
                raw=row,
            )
