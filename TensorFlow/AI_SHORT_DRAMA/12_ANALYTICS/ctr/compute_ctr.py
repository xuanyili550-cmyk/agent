"""
点击率（CTR）分析：按剧集 x 展示位（surface，如封面 cover / 预告片 trailer）统计曝光、点击与 CTR。

在流水线中的位置：12_ANALYTICS/ctr，消费与 ingest.FileEventSource 同格式的 impression / click 事件。
CTR 反映的是"封面 / 预告能不能把人吸进来"，直接用于评估 02 故事引擎产出的 hook 与 07/08 生成的封面图效果，
也是 A/B 实验（experiments/ab_framework.py）最常用的转化指标之一。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_events(path: str | Path) -> pd.DataFrame:
    """读取 JSONL 事件文件（跳过空行）为 DataFrame；CTR 只按 episode_id / surface / event_type 分组，无需解析时间。"""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def compute_ctr(events: pd.DataFrame) -> pd.DataFrame:
    """按 (episode_id, surface) 计算 impression 数、click 数与 ctr = click / impression。

    返回列固定为 episode_id / surface / impression / click / ctr，并按剧集与展示位排序，便于直接打印或写表。
    """
    # 先按三键计数，再把 event_type 展开成列（impression / click 各一列），缺失的组合补 0
    counts = events.groupby(["episode_id", "surface", "event_type"]).size().unstack(fill_value=0).reset_index()
    # 数据里若完全没有某类事件（例如全是曝光没有点击），unstack 不会生成该列，这里补上保证后续列访问不报错
    for col in ("impression", "click"):
        if col not in counts.columns:
            counts[col] = 0
    # 曝光为 0 时把分母替换成 NA，让 ctr 显示为缺失而不是抛除零错误或得到 inf
    counts["ctr"] = (counts["click"] / counts["impression"].replace(0, pd.NA)).round(4)
    return counts[["episode_id", "surface", "impression", "click", "ctr"]].sort_values(["episode_id", "surface"])


if __name__ == "__main__":
    events = load_events(Path(__file__).parent / "sample_events.jsonl")
    print(compute_ctr(events).to_string(index=False))
