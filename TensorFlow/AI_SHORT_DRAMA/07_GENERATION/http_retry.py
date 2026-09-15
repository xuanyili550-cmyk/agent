"""带指数退避的 HTTP 请求封装，所有付费 API provider 共用。

规则：
- 连接错误 / 超时 / 429 / 5xx：等 backoff * 2^n（加抖动）后重试，最多 ``retries`` 次，
  用尽后抛 TransientProviderError 交给任务层做更长周期的重试；
- 400/422（参数错、prompt 被拒）：抛 GenerationRejectedError，不重试；
- 401/403：抛 NotConfiguredError，凭证问题重试再多次也没用；
- 429 带 Retry-After 时尊重服务端给的等待时间。

流水线位置：07_GENERATION 各 API provider（Runway / Pika / ElevenLabs / sync.so）的底层 HTTP 出口。
为什么集中封装：把"HTTP 状态码 -> 项目异常类型"的映射放在一处，provider 只关心业务 payload，
任务层拿到的异常类型就能直接决定进三级重试阶梯的哪一级。
``sleep`` 作为参数注入是为了单测时可以替换成空函数，不用真的等待。
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from errors import GenerationRejectedError, NotConfiguredError, TransientProviderError  # noqa: E402

RETRY_STATUSES = {429, 500, 502, 503, 504}  # 值得重试的状态码：限流和服务端临时故障
REJECT_STATUSES = {400, 422}  # 请求本身有问题：重试无意义，应改写提示词
AUTH_STATUSES = {401, 403}  # 凭证问题：重试无意义，应检查配置


def request_with_retry(
    method: str,
    url: str,
    *,
    retries: int = 3,
    backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 30.0,
    timeout: float = 60.0,
    sleep=time.sleep,
    **kwargs: Any,
) -> requests.Response:
    """发一次 HTTP 请求，对瞬时错误按指数退避重试，把状态码翻译成项目异常。

    返回 2xx/3xx 的 Response；``retries`` 次重试仍失败抛 TransientProviderError。
    ``kwargs`` 原样透传给 ``requests.request``（json / headers / data 等）。
    """
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
        else:
            status = response.status_code
            if status < 400:
                return response
            if status in AUTH_STATUSES:
                raise NotConfiguredError(f"{url} 返回 {status}：凭证无效或无权限")
            if status in REJECT_STATUSES:
                raise GenerationRejectedError(f"{url} 返回 {status}：{response.text[:300]}")
            if status not in RETRY_STATUSES:
                # 其余 4xx（404 等）不属于任何已知类别，直接抛 requests 原生 HTTPError
                response.raise_for_status()
            last_error = requests.HTTPError(f"HTTP {status}: {response.text[:300]}")
            retry_after = response.headers.get("Retry-After")
            if retry_after and attempt < retries:
                # 服务端明确给了等待秒数就照它的来（仍受 max_backoff 封顶）；解析失败则退回指数退避
                try:
                    sleep(min(float(retry_after), max_backoff_seconds))
                    continue
                except ValueError:
                    pass
        if attempt < retries:
            # 指数退避 + 0~25% 随机抖动：避免多个 worker 同时被限流后又同时重试（惊群）
            delay = min(backoff_seconds * (2**attempt), max_backoff_seconds)
            sleep(delay + random.uniform(0, delay * 0.25))
    raise TransientProviderError(f"{method} {url} 重试 {retries} 次后仍失败：{last_error}")


# 同一个模块可能以两种名字被 import（07 内部 `from errors import ...` 走 sys.path，13_INFRA 走
# importlib "07_GENERATION.errors"）。不做别名就会出现两份类对象，except 捕获不到对方抛的异常。
import sys as _sys

# setdefault：只在该名字还没被注册时才写入，避免覆盖已经以那个名字正常 import 的模块对象
for _name in ("http_retry", "07_GENERATION.http_retry"):
    _sys.modules.setdefault(_name, _sys.modules[__name__])
