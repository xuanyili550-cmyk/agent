"""对话记忆（历史消息）层。

生产上的 LLM 应用几乎都长这样：每次请求不是只发"当前这一句"，而是把同一个会话
（context_id）之前所有的 user/assistant 轮次一起带上，模型才能记住前文；历史消息
本身要持久化（进程重启、Celery worker 换机器都不能丢）；只有当历史长度逼近模型
上下文窗口上限时才裁掉最旧的轮次（或先压缩成摘要），否则一律保留。

本模块提供三块：
1. ``estimate_tokens`` / ``context_window_for``：离线的 token 估算和模型上下文上限表；
2. ``ConversationStore`` 及三个实现（内存 / 本地 JSON 文件 / Redis）：历史消息的持久化；
3. ``ConversationMemory``：一个会话的记忆对象，负责追加轮次、算预算、触顶时裁剪/压缩。

设计上 ``ConversationMemory`` 只依赖 ``ConversationStore`` 抽象，业务代码（``BaseAgent``、
``llm_task``）只依赖 ``ConversationMemory``——和项目里其他 Provider 抽象一样，换存储后端
只改"传哪个 store 进去"这一行。
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal

Message = dict[str, str]  # {"role": "user" | "assistant", "content": str}

# --------------------------------------------------------------------------------------
# token 估算 + 模型上下文上限
# --------------------------------------------------------------------------------------

_CJK_RE = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿＀-￯]")


def estimate_tokens(text: str) -> int:
    """离线粗估 token 数：中日韩字符按 1 字 ≈ 1 token，其余按 4 字符 ≈ 1 token。

    为什么不用各家官方 tokenizer：本地免费模型 / Claude / GPT 的 tokenizer 各不相同，
    而这里只是用来判断"离上限还有多远"，估得略偏大更安全（提前一点裁剪，绝不会把
    超限请求发出去）。如果需要精确值，给 ``ConversationMemory`` 传自定义 ``token_counter``。
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return cjk + math.ceil(other / 4)


# 模型名前缀 -> 上下文窗口（token）。只列项目里实际会用到的；查不到时退回 DEFAULT_CONTEXT_WINDOW。
CONTEXT_WINDOWS: dict[str, int] = {
    "claude": 200_000,
    "gpt-4o": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5": 16_385,
    "o1": 200_000,
    "o3": 200_000,
    "o4": 200_000,
    "microsoft/phi-3.5-mini-instruct": 128_000,
    "qwen/qwen2.5": 32_768,
    "mistralai/mistral-7b-instruct": 32_768,
    "meta-llama/llama-3.1": 128_000,
    "google/gemma-2": 8_192,
    "huggingfacetb/smollm2": 8_192,
}
DEFAULT_CONTEXT_WINDOW = 8_192


def context_window_for(model: str | None) -> int:
    """按模型名前缀查上下文上限；未知模型保守地当 8k 处理。"""
    if not model:
        return DEFAULT_CONTEXT_WINDOW
    name = model.lower()
    # 最长前缀优先，避免 "gpt-4" 抢先匹配到 "gpt-4o"
    for prefix in sorted(CONTEXT_WINDOWS, key=len, reverse=True):
        if name.startswith(prefix):
            return CONTEXT_WINDOWS[prefix]
    return DEFAULT_CONTEXT_WINDOW


# --------------------------------------------------------------------------------------
# 持久化存储
# --------------------------------------------------------------------------------------


class ConversationStore(ABC):
    """历史消息的持久化抽象：按 context_id 读/写整个会话状态（消息列表 + 摘要）。"""

    @abstractmethod
    def load(self, context_id: str) -> dict | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, context_id: str, state: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, context_id: str) -> None:
        raise NotImplementedError


class InMemoryConversationStore(ConversationStore):
    """进程内字典。只适合单元测试和单进程 demo，进程退出即丢。"""

    def __init__(self) -> None:
        self._data: dict[str, dict] = {}
        self._lock = threading.Lock()

    def load(self, context_id: str) -> dict | None:
        with self._lock:
            state = self._data.get(context_id)
            return json.loads(json.dumps(state)) if state is not None else None

    def save(self, context_id: str, state: dict) -> None:
        with self._lock:
            self._data[context_id] = json.loads(json.dumps(state))

    def delete(self, context_id: str) -> None:
        with self._lock:
            self._data.pop(context_id, None)


class FileConversationStore(ConversationStore):
    """每个会话一个 JSON 文件。单机部署 / 本地开发够用；多 worker 多机器请换 Redis。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, context_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", context_id)
        return self.root / f"{safe}.json"

    def load(self, context_id: str) -> dict | None:
        path = self._path(context_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, context_id: str, state: dict) -> None:
        path = self._path(context_id)
        tmp = path.with_suffix(".json.tmp")
        with self._lock:
            # 先写临时文件再原子改名：进程中途被杀也不会留下半截 JSON
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

    def delete(self, context_id: str) -> None:
        self._path(context_id).unlink(missing_ok=True)


class RedisConversationStore(ConversationStore):
    """Redis 存储：多 worker / 多机器共享同一份会话历史。``ttl_seconds`` 为 None 表示永不过期。

    13_INFRA 的 docker-compose 已经带了 Redis（Celery broker 用的那个），生产上直接复用。
    """

    def __init__(self, url: str, ttl_seconds: int | None = None, key_prefix: str = "llm:context:") -> None:
        import redis  # 延迟 import：没装 redis 的开发环境用文件存储也能跑

        self._client = redis.Redis.from_url(url)
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix

    def _key(self, context_id: str) -> str:
        return f"{self.key_prefix}{context_id}"

    def load(self, context_id: str) -> dict | None:
        raw = self._client.get(self._key(context_id))
        return json.loads(raw) if raw else None

    def save(self, context_id: str, state: dict) -> None:
        payload = json.dumps(state, ensure_ascii=False)
        if self.ttl_seconds:
            self._client.set(self._key(context_id), payload, ex=self.ttl_seconds)
        else:
            self._client.set(self._key(context_id), payload)

    def delete(self, context_id: str) -> None:
        self._client.delete(self._key(context_id))


# --------------------------------------------------------------------------------------
# 会话记忆
# --------------------------------------------------------------------------------------

Summarizer = Callable[[list[Message]], str]
TokenCounter = Callable[[str], int]
CompactionStrategy = Literal["truncate", "summarize"]


class ConversationMemory:
    """一个会话（context_id）的长期记忆。

    规则：
    - 历史消息全部持久化到 ``store``，**永远不主动删**（完整轨迹可审计、可回放）；
    - 每次发请求前用 ``window()`` 取"能塞进模型上下文"的那一段：
      预算 = max_context_tokens - reserve_tokens - system prompt - 本次 user prompt；
      历史总量没超预算就**原样全带**，超了才从最旧的轮次开始裁；
    - 裁剪策略 ``truncate``：直接丢最旧的 user/assistant 对；
      ``summarize``：把被丢的轮次交给 ``summarizer``（通常就是同一个 LLM）压成一段摘要，
      作为第一轮历史注入，之后再触顶只会在旧摘要的基础上继续累积。
    """

    def __init__(
        self,
        context_id: str,
        store: ConversationStore | None = None,
        *,
        max_context_tokens: int = DEFAULT_CONTEXT_WINDOW,
        reserve_tokens: int = 4_096,
        token_counter: TokenCounter = estimate_tokens,
        compaction: CompactionStrategy = "truncate",
        summarizer: Summarizer | None = None,
    ) -> None:
        if compaction == "summarize" and summarizer is None:
            raise ValueError("compaction='summarize' 需要同时传 summarizer")
        self.context_id = context_id
        self.store = store or InMemoryConversationStore()
        self.max_context_tokens = max_context_tokens
        self.reserve_tokens = reserve_tokens  # 留给模型输出 + 结构化标记的余量
        self.token_counter = token_counter
        self.compaction = compaction
        self.summarizer = summarizer
        self.messages: list[Message] = []
        self.summary: str = ""  # 被压缩掉的旧轮次的摘要（仅 summarize 策略使用）
        self.summarized_upto: int = 0  # messages[:summarized_upto] 已经包含在 summary 里
        self.created_at: float = time.time()
        self.updated_at: float = self.created_at
        self.load()

    # ---- 持久化 ----------------------------------------------------------------------

    def load(self) -> None:
        state = self.store.load(self.context_id)
        if not state:
            return
        self.messages = list(state.get("messages", []))
        self.summary = state.get("summary", "")
        self.summarized_upto = int(state.get("summarized_upto", 0))
        self.created_at = float(state.get("created_at", self.created_at))
        self.updated_at = float(state.get("updated_at", self.updated_at))

    def save(self) -> None:
        self.updated_at = time.time()
        self.store.save(
            self.context_id,
            {
                "context_id": self.context_id,
                "messages": self.messages,
                "summary": self.summary,
                "summarized_upto": self.summarized_upto,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            },
        )

    def clear(self) -> None:
        self.messages = []
        self.summary = ""
        self.summarized_upto = 0
        self.store.delete(self.context_id)

    # ---- 追加轮次 --------------------------------------------------------------------

    def add_turn(self, user_prompt: str, assistant_reply: str) -> None:
        """追加一轮完整的 user -> assistant 对话并立即落盘。

        只记录最终成功的那一轮：``BaseAgent.generate()`` 里校验失败的重试属于中间过程，
        不进历史，否则历史会被无效 JSON 和报错信息灌满。
        """
        self.messages.append({"role": "user", "content": user_prompt})
        self.messages.append({"role": "assistant", "content": assistant_reply})
        self.save()

    # ---- 取窗口 ----------------------------------------------------------------------

    @property
    def turn_count(self) -> int:
        return len(self.messages) // 2

    def _tokens_of(self, messages: list[Message]) -> int:
        return sum(self.token_counter(m["content"]) for m in messages)

    def _summary_messages(self) -> list[Message]:
        if not self.summary:
            return []
        return [
            {"role": "user", "content": f"[更早的对话摘要]\n{self.summary}"},
            {"role": "assistant", "content": "好的，我已记住以上前情，后续会保持一致。"},
        ]

    def budget_for(self, system_prompt: str, user_prompt: str) -> int:
        return (
            self.max_context_tokens
            - self.reserve_tokens
            - self.token_counter(system_prompt)
            - self.token_counter(user_prompt)
        )

    def window(self, system_prompt: str = "", user_prompt: str = "") -> list[Message]:
        """返回本次请求应携带的历史消息（不含本次 user prompt）。未触顶就是全部历史。"""
        budget = self.budget_for(system_prompt, user_prompt)
        live = self.messages[self.summarized_upto :]
        prefix = self._summary_messages()
        if self._tokens_of(prefix) + self._tokens_of(live) <= budget:
            return prefix + live

        # 触顶：从最旧的轮次开始，成对（user+assistant）丢，直到剩余部分塞得下
        keep_from = 0
        while keep_from < len(live) and self._tokens_of(prefix) + self._tokens_of(live[keep_from:]) > budget:
            keep_from += 2
        dropped = live[:keep_from]
        kept = live[keep_from:]

        if self.compaction == "summarize" and dropped:
            assert self.summarizer is not None
            self.summary = self.summarizer(self._summary_messages() + dropped)
            self.summarized_upto += keep_from
            self.save()
            prefix = self._summary_messages()
            # 摘要本身也可能过长；再按预算复核一次，仍超就继续丢 kept 里最旧的
            while kept and self._tokens_of(prefix) + self._tokens_of(kept) > budget:
                kept = kept[2:]
            return prefix + kept if self._tokens_of(prefix) <= budget else kept

        # truncate：只裁本次发送的窗口，store 里的完整历史保持不动
        return kept
