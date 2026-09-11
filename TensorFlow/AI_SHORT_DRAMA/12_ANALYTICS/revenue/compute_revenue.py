from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_events(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def compute_revenue(events: pd.DataFrame) -> dict:
    impressions = events[events["event_type"] == "impression"]
    purchases = events[events["event_type"] == "unlock_purchase"].copy()
    purchases["amount_usd"] = purchases["amount_usd"].astype(float)

    total_revenue = round(purchases["amount_usd"].sum(), 4)
    total_impressions = len(impressions)
    total_users = events["user_id"].nunique()
    paying_users = purchases["user_id"].nunique()

    rpm = round(total_revenue / total_impressions * 1000, 4) if total_impressions else 0.0
    arpu = round(total_revenue / total_users, 4) if total_users else 0.0
    arppu = round(total_revenue / paying_users, 4) if paying_users else 0.0

    per_episode = (
        purchases.groupby("episode_id")["amount_usd"]
        .sum()
        .round(4)
        .rename("revenue_usd")
        .reset_index()
        .sort_values("revenue_usd", ascending=False)
    )

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
