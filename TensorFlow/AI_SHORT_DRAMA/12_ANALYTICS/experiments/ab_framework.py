"""
A/B 实验框架：确定性分桶 + 曝光 / 转化汇总 + 双比例 z 检验。

在流水线中的位置：12_ANALYTICS/experiments。短剧平台常见的实验对象是预告片钩子、封面、定价、解锁位置等；
本模块提供三件事：
1. ``assign_variant``：给用户分配实验组，基于哈希而不是随机数，保证同一用户在任何机器、任何时间都落到同一组（可复现、无需存储分配表）；
2. ``summarize_experiment``：从事件流里统计各组的曝光用户数、转化用户数与转化率；
3. ``two_proportion_z_test``：判断两组转化率差异是否显著，避免把随机波动当成效果。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.stats import norm


def assign_variant(
    user_id: str,
    experiment_name: str,
    variants: tuple[str, ...] = ("control", "treatment"),
    weights: tuple[float, ...] | None = None,
    salt: str = "",
) -> str:
    """把用户确定性地分到某个实验组。

    做法：对 "experiment_name:salt:user_id" 做 md5，取前 8 位十六进制映射到 [0, 1) 的桶值，再按各组权重的累积区间落桶。
    为什么把 experiment_name 混进哈希：同一用户在不同实验里应互相独立地分组，否则所有实验的 control 组都是同一批人，结果会相互污染。
    ``salt`` 用于需要重新洗牌（比如实验重跑）时改变分配而不改实验名。
    """
    # 未给权重时各组等分
    weights = weights or tuple(1.0 / len(variants) for _ in variants)
    digest = hashlib.md5(f"{experiment_name}:{salt}:{user_id}".encode()).hexdigest()
    # 由哈希得到 [0, 1) 内的稳定桶值，与 dict/set 的遍历顺序无关
    bucket = int(digest[:8], 16) / 0xFFFFFFFF

    # 按权重累积区间落桶：例如权重 (0.5, 0.5) -> bucket < 0.5 归 control，否则归 treatment
    cumulative = 0.0
    for variant, weight in zip(variants, weights):
        cumulative += weight
        if bucket < cumulative:
            return variant
    # 权重和因浮点误差略小于 1 时，桶值可能落到区间之外，兜底归最后一组
    return variants[-1]


def load_events(path: str | Path) -> pd.DataFrame:
    """读取 JSONL 实验事件文件（跳过空行）为 DataFrame；每行含 experiment_name / variant / event_type / user_id。"""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def summarize_experiment(events: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    """汇总指定实验各组的曝光用户数、转化用户数与转化率。

    用 ``nunique`` 按用户去重而不是数事件条数：同一用户多次曝光 / 多次转化只算一次，否则活跃用户会被高估权重。
    """
    exp = events[events["experiment_name"] == experiment_name]
    exposures = exp[exp["event_type"] == "exposure"].groupby("variant")["user_id"].nunique()
    conversions = exp[exp["event_type"] == "conversion"].groupby("variant")["user_id"].nunique()
    # 某组若一个转化都没有，conversions 里不会有该组的行，fillna(0) 补齐
    summary = pd.DataFrame({"exposed_users": exposures, "converted_users": conversions}).fillna(0)
    summary["converted_users"] = summary["converted_users"].astype(int)
    summary["conversion_rate"] = (summary["converted_users"] / summary["exposed_users"]).round(4)
    return summary.reset_index()


def two_proportion_z_test(successes_a: int, n_a: int, successes_b: int, n_b: int) -> dict:
    """双比例 z 检验（双侧）：判断 B 组转化率相对 A 组的提升是否显著。

    使用合并比例 p_pool 估计标准误（原假设下两组比例相同）；返回两组比率、绝对提升 lift、z 统计量、p 值以及是否在 0.05 水平显著。
    """
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se = (p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)) ** 0.5
    # 两组全 0 或全 1 时 se 为 0，此时无差异可言，z 取 0 避免除零
    z = (p_b - p_a) / se if se > 0 else 0.0
    # 双侧检验：p = 2 * P(Z > |z|)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return {
        "rate_a": round(p_a, 4),
        "rate_b": round(p_b, 4),
        "lift": round(p_b - p_a, 4),
        "z_stat": round(float(z), 4),
        "p_value": round(float(p_value), 6),
        "significant_at_0.05": bool(p_value < 0.05),
    }


if __name__ == "__main__":
    events = load_events(Path(__file__).parent / "sample_events.jsonl")
    summary = summarize_experiment(events, "trailer_hook_v2")
    print(summary.to_string(index=False))

    # 取 control / treatment 两行做显著性检验
    row_a = summary[summary["variant"] == "control"].iloc[0]
    row_b = summary[summary["variant"] == "treatment"].iloc[0]
    test = two_proportion_z_test(
        int(row_a["converted_users"]),
        int(row_a["exposed_users"]),
        int(row_b["converted_users"]),
        int(row_b["exposed_users"]),
    )
    print(test)
