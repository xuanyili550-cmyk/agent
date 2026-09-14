"""三级自动重试阶梯（纯逻辑，不依赖任何 provider，可单测）。

一次镜头生成失败（模型报错或 QC 不通过）后，按下面顺序升级，直到成功或阶梯用完：

    第 1 级 same_params     同样参数再试一次 —— 大概率是网络抖动 / 采样运气差（换 seed）
    第 2 级 rewrite_prompt  让 AI 改写提示词再试 —— "要不换个说法？"（措辞触发安全过滤、描述不够具体）
    第 3 级 downscale       降低分辨率再试 —— 1080p 不行就 720p、540p，总比整条流水线失败强

失败类型决定从哪一级开始：
- TransientProviderError（网络/限流）      -> 第 1 级
- QC 不通过 / GenerationRejectedError      -> 直接第 2 级（同参数再试没意义）
- ResourceExhaustedError（显存/尺寸上限）  -> 直接第 3 级

``RetryLadder.next_attempt()`` 只返回"下一次该怎么试"，真正执行由 13_INFRA/queue/shot_task.py 做。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class RetryLevel(str, Enum):
    INITIAL = "initial"
    SAME_PARAMS = "same_params"
    REWRITE_PROMPT = "rewrite_prompt"
    DOWNSCALE = "downscale"


class FailureKind(str, Enum):
    TRANSIENT = "transient"  # 网络抖动、429、5xx
    QC_REJECTED = "qc_rejected"  # 生成成功但没过质检
    PROMPT_REJECTED = "prompt_rejected"  # provider 明确拒绝了 prompt（安全过滤等）
    RESOURCE = "resource"  # 显存不足 / 分辨率超限
    UNKNOWN = "unknown"


@dataclass
class AttemptPlan:
    attempt: int
    level: RetryLevel
    width: int
    height: int
    rewrite_prompt: bool
    reason: str = ""


@dataclass
class RetryLadder:
    resolution_ladder: Sequence[tuple[int, int]] = ((1080, 1920), (720, 1280), (540, 960))
    max_attempts: int = 4  # 首次 + 三级各一次
    history: list[AttemptPlan] = field(default_factory=list)
    _resolution_index: int = 0
    _rewrites: int = 0
    _same_param_retries: int = 0

    def first_attempt(self) -> AttemptPlan:
        w, h = self.resolution_ladder[0]
        plan = AttemptPlan(attempt=1, level=RetryLevel.INITIAL, width=w, height=h, rewrite_prompt=False)
        self.history.append(plan)
        return plan

    @property
    def attempts_made(self) -> int:
        return len(self.history)

    def next_attempt(self, failure: FailureKind, reason: str = "") -> AttemptPlan | None:
        """根据这次失败的类型给出下一次尝试；阶梯用尽返回 None。"""
        if self.attempts_made >= self.max_attempts:
            return None

        level = self._pick_level(failure)
        if level is None:
            return None

        w, h = self.resolution_ladder[self._resolution_index]
        rewrite = False
        if level == RetryLevel.SAME_PARAMS:
            self._same_param_retries += 1
        elif level == RetryLevel.REWRITE_PROMPT:
            self._rewrites += 1
            rewrite = True
        elif level == RetryLevel.DOWNSCALE:
            self._resolution_index += 1
            w, h = self.resolution_ladder[self._resolution_index]

        plan = AttemptPlan(attempt=self.attempts_made + 1, level=level, width=w, height=h, rewrite_prompt=rewrite, reason=reason)
        self.history.append(plan)
        return plan

    def _pick_level(self, failure: FailureKind) -> RetryLevel | None:
        can_downscale = self._resolution_index + 1 < len(self.resolution_ladder)
        if failure == FailureKind.RESOURCE:
            return RetryLevel.DOWNSCALE if can_downscale else None
        if failure == FailureKind.TRANSIENT and self._same_param_retries == 0:
            return RetryLevel.SAME_PARAMS
        # QC 不过 / prompt 被拒 / 同参数已经试过：先改写提示词，再降分辨率
        if self._rewrites == 0:
            return RetryLevel.REWRITE_PROMPT
        if can_downscale:
            return RetryLevel.DOWNSCALE
        return None

    def summary(self) -> list[dict]:
        return [
            {"attempt": p.attempt, "level": p.level.value, "resolution": f"{p.width}x{p.height}", "rewrite_prompt": p.rewrite_prompt, "reason": p.reason}
            for p in self.history
        ]


# 同一个模块可能以两种名字被 import（07 内部 `from errors import ...` 走 sys.path，13_INFRA 走
# importlib "07_GENERATION.errors"）。不做别名就会出现两份类对象，except 捕获不到对方抛的异常。
import sys as _sys

for _name in ("retry_ladder", "07_GENERATION.retry_ladder"):
    _sys.modules.setdefault(_name, _sys.modules[__name__])
