from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import PayloadValidationError, PublishClient, PublishResult  # type: ignore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "10_EPISODES"))
from episode_manifest import EpisodeManifest, PublishStatus  # type: ignore  # noqa: E402


@dataclass
class GenericHTTPAdapterConfig:
    endpoint: str
    method: str = "POST"
    token_env_var: str = ""
    timeout_sec: int = 30
    headers: dict[str, str] = field(default_factory=dict)
    payload_template: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GenericHTTPAdapterConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            endpoint=raw["endpoint"],
            method=raw.get("method", "POST"),
            token_env_var=raw.get("token_env_var", ""),
            timeout_sec=raw.get("timeout_sec", 30),
            headers=raw.get("headers", {}) or {},
            payload_template=raw.get("payload_template", {}) or {},
        )


class GenericHTTPAdapter(PublishClient):
    """Configurable HTTP upload adapter for platforms with no public developer API.

    This is intentionally generic: endpoint, headers and payload shape are all
    supplied via a YAML config (or overridden in __init__), because there is no
    official spec to code against. See the README next to each platform's
    config for how real integration would have to be reverse-engineered or
    negotiated with the platform directly.
    """

    def __init__(self, platform_name: str, config: GenericHTTPAdapterConfig) -> None:
        self.platform_name = platform_name
        self.config = config

    def _substitute(self, template_value: Any, context: dict[str, str]) -> Any:
        if isinstance(template_value, str):
            return Template(template_value).safe_substitute(context)
        if isinstance(template_value, dict):
            return {k: self._substitute(v, context) for k, v in template_value.items()}
        if isinstance(template_value, list):
            return [self._substitute(v, context) for v in template_value]
        return template_value

    def build_payload(self, episode: EpisodeManifest) -> dict[str, Any]:
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
        if not self.config.endpoint or "REPLACE_WITH" in self.config.endpoint:
            raise PayloadValidationError(f"{self.platform_name}: endpoint is not configured (edit config.yaml or set an env var override)")
        if not payload:
            raise PayloadValidationError(f"{self.platform_name}: empty payload")
        for key, value in payload.items():
            if isinstance(value, str) and value.startswith("$"):
                raise PayloadValidationError(f"{self.platform_name}: unresolved template placeholder for field '{key}' -> '{value}'")

    def _do_upload(self, episode: EpisodeManifest, payload: dict[str, Any]) -> PublishResult:
        headers = dict(self.config.headers)
        if self.config.token_env_var:
            token = os.environ.get(self.config.token_env_var, "")
            if not token:
                raise RuntimeError(f"{self.config.token_env_var} is not set; cannot call {self.platform_name} upload endpoint")
            headers.setdefault("Authorization", f"Bearer {token}")

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
                error_message=f"HTTP {resp.status_code}: {resp.text[:500]}",
                payload=payload,
            )
        data: dict[str, Any] = {}
        try:
            data = resp.json()
        except ValueError:
            pass
        return PublishResult(
            platform=self.platform_name,
            episode_id=episode.episode_id,
            status=PublishStatus.PENDING,
            remote_id=data.get("id") or data.get("episode_id"),
            remote_url=data.get("url"),
            payload=payload,
        )
