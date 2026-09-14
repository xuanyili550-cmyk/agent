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
        self.path = Path(path)

    def fetch(self, since: Optional[datetime] = None) -> Iterator[AnalyticsItem]:
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
                amount_usd=float(row["amount_usd"]) if row.get("amount_usd") is not None else None,
                experiment=row.get("experiment"),
                variant=row.get("variant"),
                raw=row,
            )
