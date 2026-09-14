from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generic_http_adapter import GenericHTTPAdapter, GenericHTTPAdapterConfig  # type: ignore  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
EXAMPLE_CONFIG_PATH = Path(__file__).resolve().parent / "config.example.yaml"


class DramaBoxPublishClient(GenericHTTPAdapter):
    """See README.md: DramaBox has no public developer API. This is a
    configurable HTTP skeleton, not a real integration."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        resolved = Path(
            config_path or os.environ.get("DRAMABOX_CONFIG_PATH", "") or (DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else EXAMPLE_CONFIG_PATH)
        )
        super().__init__("dramabox", GenericHTTPAdapterConfig.from_yaml(resolved))
