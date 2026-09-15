"""GoodShort 发布客户端：GenericHTTPAdapter 的薄封装，只负责定位配置文件。

GoodShort 没有公开开发者 API，所以端点/请求体全靠 config.yaml；配置查找顺序：
显式参数 > GOODSHORT_CONFIG_PATH 环境变量 > 同目录 config.yaml > config.example.yaml（示例里端点是 REPLACE_WITH 占位，
validate_payload 会拦下来，因此没配置时 dry-run 会给出明确的"endpoint is not configured"提示而不是误发请求）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# parents[1] 是 11_PUBLISH，从那里 import 通用适配器
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generic_http_adapter import GenericHTTPAdapter, GenericHTTPAdapterConfig  # type: ignore  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
EXAMPLE_CONFIG_PATH = Path(__file__).resolve().parent / "config.example.yaml"


class GoodShortPublishClient(GenericHTTPAdapter):
    """见 README.md：GoodShort 没有公开开发者 API。这是一个可配置的 HTTP 骨架，不是真正的接入实现。"""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """按"参数 > 环境变量 > config.yaml > config.example.yaml"的顺序解析配置路径并加载。"""
        resolved = Path(
            config_path or os.environ.get("GOODSHORT_CONFIG_PATH", "") or (DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else EXAMPLE_CONFIG_PATH)
        )
        super().__init__("goodshort", GenericHTTPAdapterConfig.from_yaml(resolved))
