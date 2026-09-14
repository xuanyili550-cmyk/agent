from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEFAULT_RETENTION_DAYS = (1, 7, 30)


def load_events(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["event_date"] = df["timestamp"].dt.floor("D")
    return df


def compute_retention(events: pd.DataFrame, retention_days: tuple[int, ...] = DEFAULT_RETENTION_DAYS) -> pd.DataFrame:
    cohort_date = events.groupby("user_id")["event_date"].min().rename("cohort_date")
    activity_dates = events.groupby("user_id")["event_date"].apply(set)

    cohort_size = len(cohort_date)
    rows = []
    for day in retention_days:
        retained = 0
        for user_id, c_date in cohort_date.items():
            target_date = c_date + pd.Timedelta(days=day)
            if target_date in activity_dates.loc[user_id]:
                retained += 1
        rate = retained / cohort_size if cohort_size else 0.0
        rows.append(
            {
                "day": day,
                "cohort_size": cohort_size,
                "retained_users": retained,
                "retention_rate": round(rate, 4),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    events = load_events(Path(__file__).parent / "sample_events.jsonl")
    print(compute_retention(events).to_string(index=False))
