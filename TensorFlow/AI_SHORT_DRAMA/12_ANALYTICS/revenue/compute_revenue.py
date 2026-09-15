"""
收入分析：基于曝光与付费解锁事件计算总收入、RPM、ARPU、ARPPU 及分剧集收入。

在流水线中的位置：12_ANALYTICS/revenue，消费与 ingest.FileEventSource 同格式的 impression / unlock_purchase 事件。
短剧的商业模式是"前几集免费、后续按集付费解锁"（见 10_EPISODES 的 is_free / unlock_price_credits），
所以按剧集拆收入能直接看出付费墙放在哪一集效果最好。

指标含义：
- RPM   = 每千次曝光收入，衡量流量变现效率；
- ARPU  = 总收入 / 全部用户数，衡量整体用户价值；
- ARPPU = 总收入 / 付费用户数，衡量付费用户的客单价；ARPU 与 ARPPU 的差距反映付费转化率的高低。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_events(path: str | Path) -> pd.DataFrame:
    """读取 JSONL 事件文件（跳过空行）为 DataFrame。"""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def compute_revenue(events: pd.DataFrame) -> dict:
    """计算收入指标，返回包含总量、RPM / ARPU / ARPPU 与 ``revenue_by_episode``（DataFrame）的字典。

    所有比率在分母为 0 时返回 0.0 而不是抛错，保证空数据 / 冷启动阶段也能出报表。
    """
    impressions = events[events["event_type"] == "impression"]
    # copy() 避免后面给切片赋值触发 pandas 的 SettingWithCopyWarning
    purchases = events[events["event_type"] == "unlock_purchase"].copy()
    purchases["amount_usd"] = purchases["amount_usd"].astype(float)

    total_revenue = round(purchases["amount_usd"].sum(), 4)
    total_impressions = len(impressions)
    # 分母用去重用户数：同一用户多次曝光 / 多次购买不应重复计入
    total_users = events["user_id"].nunique()
    paying_users = purchases["user_id"].nunique()

    rpm = round(total_revenue / total_impressions * 1000, 4) if total_impressions else 0.0
    arpu = round(total_revenue / total_users, 4) if total_users else 0.0
    arppu = round(total_revenue / paying_users, 4) if paying_users else 0.0

    # 按剧集汇总收入并降序，第一眼就能看到最能变现的那几集
    per_episode = purchases.groupby("episode_id")["amount_usd"].sum().round(4).rename("revenue_usd").reset_index().sort_values("revenue_usd", ascending=False)

    return {
        "total_revenue_usd": total_revenue,
        "total_impressions": total_impressions,
        "total_users": total_users,
        "paying_users": paying_users,
        "rpm": rpm,
        "arpu": arpu,
        "arppu": arppu,
        "revenue_by_episode": per_episode,
    }


if __name__ == "__main__":
    events = load_events(Path(__file__).parent / "sample_events.jsonl")
    result = compute_revenue(events)
    for key, value in result.items():
        if key == "revenue_by_episode":
            print("revenue_by_episode:")
            print(value.to_string(index=False))
        else:
            print(f"{key}: {value}")
