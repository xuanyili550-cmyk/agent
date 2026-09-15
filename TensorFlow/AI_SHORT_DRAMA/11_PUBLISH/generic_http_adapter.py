"""通用 HTTP 发布适配器：给没有公开开发者 API 的平台（ReelShort / DramaBox / GoodShort）用的可配置上传骨架。

这些平台没有官方接口文档可依，所以端点、请求头、请求体形状全部来自每个平台目录下的 YAML 配置，
代码本身只负责"模板替换 + 发请求 + 解析响应"。真正接入时只需改 config.yaml，不用改代码。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

import requests
import yaml

# 先把本目录加进 sys.path 以便 import base；再加 10_EPISODES（目录名以数字开头，不能直接 import）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import PayloadValidationError, PublishClient, PublishResult  # type: ignore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "10_EPISODES"))
from episode_manifest import EpisodeManifest, PublishStatus  # type: ignore  # noqa: E402


@dataclass
class GenericHTTPAdapterConfig:
    """一个平台的 HTTP 上传配置：端点、方法、承载 token 的环境变量名、超时、额外请求头、请求体模板。

    payload_template 的值是 string.Template 语法（$episode_id、$title ...），在 build_payload 时用 manifest 字段填充。
    """

    endpoint: str
    method: str = "POST"
    token_env_var: str = ""
    timeout_sec: int = 30
    headers: dict[str, str] = field(default_factory=dict)
    payload_template: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GenericHTTPAdapterConfig:
        """从 YAML 文件加载配置。除 endpoint 外都有默认值；空文件当作空 dict 处理。"""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            endpoint=raw["endpoint"],
            method=raw.get("method", "POST"),
            token_env_var=raw.get("token_env_var", ""),
            timeout_sec=raw.get("timeout_sec", 30),
            headers=raw.get("headers", {}) or {},  # YAML 里写了 key 但值为空会得到 None，这里兜底成 {}
            payload_template=raw.get("payload_template", {}) or {},
        )


class GenericHTTPAdapter(PublishClient):
    """给没有公开开发者 API 的平台用的可配置 HTTP 上传适配器。

    刻意做得很通用：端点、请求头、请求体形状都由 YAML 配置提供（或在 __init__ 里覆盖），
    因为没有官方规范可以照着写。每个平台配置旁边的 README 说明了真实接入需要如何逆向或与平台直接协商。
    """

    def __init__(self, platform_name: str, config: GenericHTTPAdapterConfig) -> None:
        """记录平台名（写进 PublishResult）和配置。"""
        self.platform_name = platform_name
        self.config = config

    def _substitute(self, template_value: Any, context: dict[str, str]) -> Any:
        """递归地对模板做变量替换：字符串走 Template.safe_substitute，dict/list 逐项递归，其他类型原样返回。

        用 safe_substitute 而不是 substitute：缺变量时保留 $xxx 原样，交给 validate_payload 统一报错，而不是在这里炸掉。
        """
        if isinstance(template_value, str):
            return Template(template_value).safe_substitute(context)
        if isinstance(template_value, dict):
            return {k: self._substitute(v, context) for k, v in template_value.items()}
        if isinstance(template_value, list):
            return [self._substitute(v, context) for v in template_value]
        return template_value

    def build_payload(self, episode: EpisodeManifest) -> dict[str, Any]:
        """用 manifest 字段 + 大写环境变量填充 payload_template。

        环境变量用 setdefault 加入，所以 manifest 字段优先；只取全大写的 key 是为了避免把 PATH/HOME 之外的杂项也当模板变量。
        """
        context = {
            "episode_id": episode.episode_id,
            "series_id": episode.series_id,
            "episode_number": str(episode.episode_number),
            "title": episode.title,
            "video_path": episode.video_path,
            "subtitle_path": episode.subtitle_path or "",
            "duration_sec": str(episode.duration_sec),
            "language": episode.language,
        }
        for key in os.environ:
            if key.isupper():
                context.setdefault(key, os.environ[key])
        return self._substitute(self.config.payload_template, context)

    def validate_payload(self, payload: dict[str, Any]) -> None:
        """本地校验：端点已配置（不是示例里的 REPLACE_WITH 占位）、请求体非空、顶层字段没有残留的 $ 占位符。"""
        if not self.config.endpoint or "REPLACE_WITH" in self.config.endpoint:
            raise PayloadValidationError(f"{self.platform_name}: endpoint is not configured (edit config.yaml or set an env var override)")
        if not payload:
            raise PayloadValidationError(f"{self.platform_name}: empty payload")
        for key, value in payload.items():
            if isinstance(value, str) and value.startswith("$"):
                raise PayloadValidationError(f"{self.platform_name}: unresolved template placeholder for field '{key}' -> '{value}'")

    def _do_upload(self, episode: EpisodeManifest, payload: dict[str, Any]) -> PublishResult:
        """发起 HTTP 请求。token 从环境变量读（不落盘）；HTTP >= 400 记 FAILED，否则记 PENDING 并尽量从响应里取 id/url。"""
        headers = dict(self.config.headers)
        if self.config.token_env_var:
            token = os.environ.get(self.config.token_env_var, "")
            if not token:
                raise RuntimeError(f"{self.config.token_env_var} is not set; cannot call {self.platform_name} upload endpoint")
            headers.setdefault("Authorization", f"Bearer {token}")  # 配置里显式写了 Authorization 就不覆盖

        resp = requests.request(
            self.config.method,
            self.config.endpoint,
            json=payload,
            headers=headers,
            timeout=self.config.timeout_sec,
        )
        if resp.status_code >= 400:
            return PublishResult(
                platform=self.platform_name,
                episode_id=episode.episode_id,
                status=PublishStatus.FAILED,
                error_message=f"HTTP {resp.status_code}: {resp.text[:500]}",  # 截断，避免把整页 HTML 错误页塞进数据库
                payload=payload,
            )
        data: dict[str, Any] = {}
        try:
            data = resp.json()
        except ValueError:
            pass  # 响应不是 JSON 也算成功提交，只是拿不到 remote_id
        # 没有官方规范，猜测常见字段名；状态记 PENDING 是因为这些平台没有状态接口，无法确认最终发布结果
        return PublishResult(
            platform=self.platform_name,
            episode_id=episode.episode_id,
            status=PublishStatus.PENDING,
            remote_id=data.get("id") or data.get("episode_id"),
            remote_url=data.get("url"),
            payload=payload,
        )
