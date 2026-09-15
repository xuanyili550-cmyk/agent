"""
12_ANALYTICS 演示脚本：依次运行留存 / CTR / 收入 / A/B 实验四个子模块，各自读取自己目录下的合成 sample_events.jsonl 并打印结果。

在流水线中的位置：12_ANALYTICS 是发行之后的"数据回流"层，本脚本用来一键验证各分析模块能在无外部依赖（无数据库、无平台 API）
的情况下跑通，也是阅读各模块输出格式的最快方式。用法：python demo.py

为什么用 importlib 按文件路径加载：12_ANALYTICS 及其子目录不是常规 Python 包（顶层目录名带数字前缀、子目录无 __init__），
按路径加载可以避免为了演示而改动包结构。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load(name: str, rel_path: str):
    """按相对路径把一个 .py 文件加载成模块并注册到 ``sys.modules``（注册是为了让模块内的 dataclass / pickle 等能按名字找到它）。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def section(title: str) -> None:
    """打印一个带分隔线的小节标题，便于在终端里区分各模块输出。"""
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> None:
    """依次演示留存、CTR、收入、A/B 实验四个模块，并展示 assign_variant 的确定性分桶。"""
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
    # revenue_by_episode 是 DataFrame，单独用表格形式打印；其余都是标量
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
    # 从汇总表里取出 control / treatment 两行，做双比例 z 检验
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
