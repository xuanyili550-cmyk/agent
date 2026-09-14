from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_events(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def compute_ctr(events: pd.DataFrame) -> pd.DataFrame:
    counts = events.groupby(["episode_id", "surface", "event_type"]).size().unstack(fill_value=0).reset_index()
    for col in ("impression", "click"):
        if col not in counts.columns:
            counts[col] = 0
    counts["ctr"] = (counts["click"] / counts["impression"].replace(0, pd.NA)).round(4)
    return counts[["episode_id", "surface", "impression", "click", "ctr"]].sort_values(["episode_id", "surface"])


if __name__ == "__main__":
    events = load_events(Path(__file__).parent / "sample_events.jsonl")
    print(compute_ctr(events).to_string(index=False))
