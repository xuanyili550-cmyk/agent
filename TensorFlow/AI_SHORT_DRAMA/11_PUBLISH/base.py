"""发布客户端的公共抽象：PublishClient 基类、PublishResult 结果对象、PayloadValidationError。

11_PUBLISH 位于 10_EPISODES（成片 + manifest）之后，是流水线的最后一环：把一集成片推到各平台。
所有平台客户端都走同一条模板方法 upload()：build_payload -> validate_payload -> (dry_run 直接返回 / 真上传)。
这样设计是为了让"组装请求体 + 本地校验"不依赖网络和真实凭据就能被测试和 CLI 演练，真正的网络调用被隔离在 _do_upload 里。
"""

from __future__ import annotations

import abc
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# 10_EPISODES 目录名以数字开头不能直接 import，把它塞进 sys.path 后按模块名引用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "10_EPISODES"))
from episode_manifest import EpisodeManifest, PublishStatus  # type: ignore  # noqa: E402


class PayloadValidationError(ValueError):
    """请求体不合平台要求（标题超长、缺字段、模板占位符没替换等）。在发起网络请求之前就抛出。"""

    pass


@dataclass
class PublishResult:
    """一次发布尝试的结果。

    status 沿用 10_EPISODES 的 PublishStatus；remote_id 是平台侧的内容 ID（后续 fetch_status 轮询要用）；
    dry_run=True 时 payload 会带回来供 CLI/测试检查；requested_at 用 UTC，避免多机时区不一致。
    """

    platform: str
    episode_id: str
    status: PublishStatus
    remote_id: str | None = None
    remote_url: str | None = None
    dry_run: bool = False
    payload: dict[str, Any] | None = None
    error_message: str | None = None
    requested_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class PublishClient(abc.ABC):
    """平台发布客户端基类。子类必须实现 build_payload / validate_payload / _do_upload，可选覆盖 fetch_status。"""

    platform_name: str

    @abc.abstractmethod
    def build_payload(self, episode: EpisodeManifest) -> dict[str, Any]:
        """把 EpisodeManifest 转成平台特定的请求体。"""

    @abc.abstractmethod
    def validate_payload(self, payload: dict[str, Any]) -> None:
        """请求体不合规时抛 PayloadValidationError。不得访问网络。"""

    @abc.abstractmethod
    def _do_upload(self, episode: EpisodeManifest, payload: dict[str, Any]) -> PublishResult:
        """执行真正的网络调用。只在 dry_run=False 时被调用。"""

    def fetch_status(self, remote_id: str) -> tuple[PublishStatus, str | None, str | None]:
        """查询已提交内容在平台侧的处理状态：(status, remote_url, error_message)。

        上传成功不等于发布成功（YouTube 要转码、TikTok 要审核），13_INFRA 的
        publish_status_task 会周期性调这个方法把最终状态回写到 manifest 和数据库。
        没有状态接口的平台保持默认实现：返回 PENDING 表示"无法确认"。
        """
        return PublishStatus.PENDING, None, "platform does not expose a status endpoint"

    def upload(self, episode: EpisodeManifest, dry_run: bool = False) -> PublishResult:
        """模板方法：组装请求体 -> 本地校验 -> dry_run 则原样返回请求体，否则真上传。

        校验放在 dry_run 判断之前，这样 dry-run 也能暴露配置/字段问题，这正是 publish_cli --dry-run 的用途。
        """
        payload = self.build_payload(episode)
        self.validate_payload(payload)
        if dry_run:
            return PublishResult(
                platform=self.platform_name,
                episode_id=episode.episode_id,
                status=PublishStatus.PENDING,
                dry_run=True,
                payload=payload,
            )
        return self._do_upload(episode, payload)
