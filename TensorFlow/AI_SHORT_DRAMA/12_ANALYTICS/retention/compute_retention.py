"""
留存分析：基于用户级事件计算 D1 / D7 / D30 留存率。

在流水线中的位置：12_ANALYTICS/retention，消费 ``ingest.FileEventSource`` 同格式的用户事件（每行含 user_id、timestamp），
产出的留存率是判断一部剧"能不能留住人"的核心指标，反过来指导 02 故事引擎的钩子 / 悬念策略。

为什么按 cohort（首次活跃日）算：留存的定义是"第一天来的人，第 N 天还在不在"。每个用户的"第 0 天"不同，
必须先按各自首次出现的日期定锚，再看其 +N 天是否有活动；直接用"第 N 天的活跃数 / 总用户数"会把新老用户混在一起，得不到可比的数字。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# 短剧行业惯用的三个观察窗口：次日、7 日、30 日
DEFAULT_RETENTION_DAYS = (1, 7, 30)


def load_events(path: str | Path) -> pd.DataFrame:
    """读取 JSONL 事件文件为 DataFrame，把 timestamp 解析成 UTC 时间并向下取整到"天"（``event_date``），留存只关心日粒度。"""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["event_date"] = df["timestamp"].dt.floor("D")
    return df


def compute_retention(events: pd.DataFrame, retention_days: tuple[int, ...] = DEFAULT_RETENTION_DAYS) -> pd.DataFrame:
    """计算每个观察窗口的留存：以每个用户的首次活跃日为 cohort 起点，统计第 +day 天仍有活动的用户占比。

    返回列：day / cohort_size / retained_users / retention_rate。这里把所有用户当作一个整体 cohort（cohort_size 为全体用户数），
    适合样本数据与单剧集视角；要看分日 cohort 曲线可在此基础上再按 cohort_date 分组。
    """
    # 每个用户的首次活跃日 = 该用户的 cohort 起点
    cohort_date = events.groupby("user_id")["event_date"].min().rename("cohort_date")
    # 每个用户所有活跃日的集合，用 set 让"第 N 天是否活跃"的判断是 O(1)
    activity_dates = events.groupby("user_id")["event_date"].apply(set)

    cohort_size = len(cohort_date)
    rows = []
    for day in retention_days:
        retained = 0
        for user_id, c_date in cohort_date.items():
            target_date = c_date + pd.Timedelta(days=day)
            if target_date in activity_dates.loc[user_id]:
                retained += 1
        # 没有用户时避免除零
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
