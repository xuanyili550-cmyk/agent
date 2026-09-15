"""平台数据接入。

在流水线中的位置：12_ANALYTICS 的入口层，把不同来源的数据统一成 ``AnalyticsItem`` 后交给留存 / CTR / 收入 / A/B 各模块，
以及 13_INFRA 的入库任务。

- ``FileEventSource``：自家 App / SDK 导出的 jsonl 用户级事件（和 12_ANALYTICS/*/sample_events.jsonl 同格式）；
- ``YouTubeAnalyticsSource``：YouTube Analytics API v2，按天/按视频聚合指标；
- ``TikTokAnalyticsSource``：TikTok Display API 的 video/query，拿视频级累计指标。
``build_source(spec)`` 按 {"name": ..., "type": ...} 构造。

本包导出：AnalyticsItem、AnalyticsSource、三个具体数据源类、build_source 与 SOURCE_TYPES 注册表。
"""

from .base import AnalyticsItem, AnalyticsSource
from .file_source import FileEventSource
from .tiktok_source import TikTokAnalyticsSource
from .youtube_source import YouTubeAnalyticsSource

# 数据源类型名 -> 实现类；配置文件里用字符串指定类型，这里做查表
SOURCE_TYPES = {
    "file": FileEventSource,
    "youtube": YouTubeAnalyticsSource,
    "tiktok": TikTokAnalyticsSource,
}


def build_source(spec: dict) -> AnalyticsSource:
    """根据配置字典构造数据源：``type`` 缺省时用 ``name`` 当类型；除 name/type 之外的键原样作为构造参数透传。"""
    kind = spec.get("type") or spec["name"]
    if kind not in SOURCE_TYPES:
        raise ValueError(f"未知数据源类型 {kind}，可选 {sorted(SOURCE_TYPES)}")
    kwargs = {k: v for k, v in spec.items() if k not in ("name", "type")}
    return SOURCE_TYPES[kind](**kwargs)


__all__ = ["AnalyticsItem", "AnalyticsSource", "FileEventSource", "YouTubeAnalyticsSource", "TikTokAnalyticsSource", "build_source", "SOURCE_TYPES"]
