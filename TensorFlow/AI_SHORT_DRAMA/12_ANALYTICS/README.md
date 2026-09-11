# 12_ANALYTICS

- `retention/compute_retention.py`：读 `user_id, episode_id, event_type, timestamp` 格式的 JSONL 事件，
  按用户首次事件日期分 cohort，计算 D1/D7/D30 留存率。
- `ctr/compute_ctr.py`：读 impression/click 事件，按 `episode_id + surface`（封面/预告）计算 CTR。
- `revenue/compute_revenue.py`：读 impression/unlock_purchase 事件，计算总收入、RPM、ARPU、ARPPU，
  以及按剧集拆分的收入。
- `experiments/ab_framework.py`：`assign_variant()` 用 md5(experiment_name:salt:user_id) 做稳定哈希分桶；
  `summarize_experiment()` 汇总 exposure/conversion；`two_proportion_z_test()` 做两比例 z 检验判断显著性。

每个子模块目录下都有一份合成的 `sample_events.jsonl`（纯编造数据，用固定随机种子生成，仅用于验证计算逻辑）。

跑 `python demo.py` 会依次跑通以上四个模块并打印结果。
