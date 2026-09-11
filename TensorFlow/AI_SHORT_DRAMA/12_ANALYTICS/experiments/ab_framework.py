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
    weights = weights or tuple(1.0 / len(variants) for _ in variants)
    digest = hashlib.md5(f"{experiment_name}:{salt}:{user_id}".encode("utf-8")).hexdigest()
    # stable bucket in [0, 1) derived from the hash, independent of dict/set ordering
    bucket = int(digest[:8], 16) / 0xFFFFFFFF

    cumulative = 0.0
    for variant, weight in zip(variants, weights):
        cumulative += weight
        if bucket < cumulative:
            return variant
    return variants[-1]


def load_events(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def summarize_experiment(events: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    exp = events[events["experiment_name"] == experiment_name]
    exposures = exp[exp["event_type"] == "exposure"].groupby("variant")["user_id"].nunique()
    conversions = exp[exp["event_type"] == "conversion"].groupby("variant")["user_id"].nunique()
    summary = pd.DataFrame({"exposed_users": exposures, "converted_users": conversions}).fillna(0)
    summary["converted_users"] = summary["converted_users"].astype(int)
    summary["conversion_rate"] = (summary["converted_users"] / summary["exposed_users"]).round(4)
    return summary.reset_index()


def two_proportion_z_test(
    successes_a: int, n_a: int, successes_b: int, n_b: int
) -> dict:
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se = (p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)) ** 0.5
    z = (p_b - p_a) / se if se > 0 else 0.0
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

    row_a = summary[summary["variant"] == "control"].iloc[0]
    row_b = summary[summary["variant"] == "treatment"].iloc[0]
    test = two_proportion_z_test(
        int(row_a["converted_users"]), int(row_a["exposed_users"]),
        int(row_b["converted_users"]), int(row_b["exposed_users"]),
    )
    print(test)
