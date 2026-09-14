# Runs every analytics submodule against its own synthetic sample_events.jsonl
# and prints the results. Usage: python demo.py
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> None:
    retention_mod = _load("retention_mod", "retention/compute_retention.py")
    ctr_mod = _load("ctr_mod", "ctr/compute_ctr.py")
    revenue_mod = _load("revenue_mod", "revenue/compute_revenue.py")
    ab_mod = _load("ab_mod", "experiments/ab_framework.py")

    section("Retention (D1/D7/D30)")
    retention_events = retention_mod.load_events(ROOT / "retention" / "sample_events.jsonl")
    print(retention_mod.compute_retention(retention_events).to_string(index=False))

    section("CTR (cover/trailer)")
    ctr_events = ctr_mod.load_events(ROOT / "ctr" / "sample_events.jsonl")
    print(ctr_mod.compute_ctr(ctr_events).to_string(index=False))

    section("Revenue (RPM/ARPU/ARPPU)")
    revenue_events = revenue_mod.load_events(ROOT / "revenue" / "sample_events.jsonl")
    revenue_result = revenue_mod.compute_revenue(revenue_events)
    for key, value in revenue_result.items():
        if key == "revenue_by_episode":
            print("revenue_by_episode:")
            print(value.to_string(index=False))
        else:
            print(f"{key}: {value}")

    section("A/B experiment (trailer_hook_v2)")
    ab_events = ab_mod.load_events(ROOT / "experiments" / "sample_events.jsonl")
    ab_summary = ab_mod.summarize_experiment(ab_events, "trailer_hook_v2")
    print(ab_summary.to_string(index=False))
    row_a = ab_summary[ab_summary["variant"] == "control"].iloc[0]
    row_b = ab_summary[ab_summary["variant"] == "treatment"].iloc[0]
    test_result = ab_mod.two_proportion_z_test(
        int(row_a["converted_users"]),
        int(row_a["exposed_users"]),
        int(row_b["converted_users"]),
        int(row_b["exposed_users"]),
    )
    print(test_result)

    example_bucket = ab_mod.assign_variant("u001", "trailer_hook_v2")
    print(f"\nassign_variant('u001', 'trailer_hook_v2') -> {example_bucket}")


if __name__ == "__main__":
    main()
