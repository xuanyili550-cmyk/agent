"""LangGraph 检查点持久化（Persistence / Checkpointer）。

给 ``graph.compile(checkpointer=...)`` 传一个 checkpointer 之后，LangGraph 会在**每个节点执行完**
自动把当前 state 存一份，并用 ``config={"configurable": {"thread_id": ...}}`` 区分不同会话：

- 同一个 ``thread_id`` 再次 invoke：从最后一个检查点接着跑，已经完成的节点不会重算；
- 不同 ``thread_id`` 之间完全隔离，互不干扰——这是多用户 / 多任务并发的基础。

对本项目的直接价值：故事阶段（story_bible -> season_arc -> characters -> episodes -> storyboard
-> prompts）每个节点都是真金白银的 LLM 调用。以前 worker 在 storyboard 节点崩了，Celery 重试会从
第一个节点重新生成，前面几十次调用白花；接上 checkpointer 后重试直接从 storyboard 继续。

本模块提供两种 checkpointer：

1. ``StoreCheckpointSaver``：把检查点写进本项目已有的 ``ConversationStore`` 抽象
   （本地 JSON 文件 / SQLite / Redis 三选一，由 ``LLM_CONTEXT_*`` 决定）。
   **为什么不直接用 langgraph-checkpoint-sqlite**：那是个额外依赖，而且会引入第二套存储配置；
   复用 ConversationStore 意味着"会话历史存哪儿，检查点就存哪儿"，部署时只配一处。
2. ``build_checkpointer``：按 store 构造上面这个 saver；传 None 时退回 LangGraph 自带的
   ``InMemorySaver``（仅单进程有效，给测试和本地 demo 用）。

和 ``agents/memory.py::ConversationMemory`` 的关系：两者并存、互为备份，不互相替代。
checkpointer 存的是**图的执行状态**（跑到哪个节点、各节点产出的强类型对象），
ConversationMemory 存的是**发给模型的对话历史**（user/assistant 轮次，可审计可回放）。
前者丢了最多重跑几个节点，后者丢了就没法解释"当初模型看到的是什么"。
"""

from __future__ import annotations

import base64
import time
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

KEY_PREFIX = "checkpoint:"


def allowed_checkpoint_types() -> list[type]:
    """从检查点还原对象时，允许重建哪些类——即 03_STRUCTURED_DATA/schemas.py 里的全部模型和枚举。

    为什么必须显式登记：LangGraph 的 allowlist 要的是精确的 ``(模块, 类名)``，**模块前缀不算匹配**。
    不登记时行为是"打一行 Deserializing unregistered type 警告并降级"——pydantic 对象会变成
    普通 dict，续跑时 ``script.content`` 直接 AttributeError；未来版本还会直接拒绝。

    为什么不图省事传 ``allowed_msgpack_modules=True``：那等于允许从检查点字节里重建任意类。
    检查点可能存在共享 Redis 里，能写 Redis 的人就能触发任意类的反序列化，这是条执行面。
    这里按模块枚举实类，schemas 加了新模型会自动包含进来，不需要手工维护清单。
    """
    import importlib
    import inspect

    schemas = importlib.import_module("schemas")
    return [obj for _name, obj in inspect.getmembers(schemas, inspect.isclass) if obj.__module__ == schemas.__name__]


def _serializer() -> JsonPlusSerializer:
    """两种 checkpointer 共用的序列化器：登记了 schemas 的全部类。"""
    return JsonPlusSerializer(allowed_msgpack_modules=allowed_checkpoint_types())


def _encode(serde: JsonPlusSerializer, value: Any) -> list[str]:
    """用 LangGraph 的 serde 把对象序列化成 ``[类型, base64 字符串]``。

    为什么要 base64：serde 给出的是 bytes，而 ConversationStore 的三种实现（JSON 文件 / SQLite
    文本列 / Redis 字符串）存的都是 JSON，塞不进裸 bytes。
    """
    type_name, payload = serde.dumps_typed(value)
    return [type_name, base64.b64encode(payload).decode("ascii")]


def _decode(serde: JsonPlusSerializer, raw: Sequence[str]) -> Any:
    """``_encode`` 的逆操作。"""
    type_name, payload = raw
    return serde.loads_typed((type_name, base64.b64decode(payload.encode("ascii"))))


class StoreCheckpointSaver(BaseCheckpointSaver):
    """把 LangGraph 检查点持久化到 ``ConversationStore``（文件 / SQLite / Redis）的 saver。

    存储布局（每个 thread_id 一条记录）::

        {
          "thread_id": "run:<id>",
          "namespaces": {
            "<checkpoint_ns>": {
              "order": ["<checkpoint_id>", ...],          # 按写入顺序，最后一个最新
              "checkpoints": {"<id>": {"checkpoint": [..], "metadata": [..], "parent": "<id>|null", "ts": 1.0}},
              "writes": {"<id>": [{"task_id":..,"task_path":..,"idx":..,"channel":..,"value":[..]}]}
            }
          }
        }

    ``checkpoint_ns``（命名空间）是 LangGraph 给子图用的，顶层图固定是空串；这里照实分开存，
    子图状态不会和主图混在一起。
    """

    def __init__(self, store: Any, *, lock_timeout: float = 60.0, max_checkpoints: int = 64) -> None:
        """``store``：任意 ConversationStore 实现。``max_checkpoints``：每个命名空间最多保留多少个
        检查点（超过就丢最旧的）——一次故事流水线只有几个节点，留 64 个足够回溯，又不会让记录无限膨胀。"""
        super().__init__(serde=_serializer())
        self.store = store
        self.lock_timeout = lock_timeout
        self.max_checkpoints = max_checkpoints

    # ---- 内部：读写整条 thread 记录 ----------------------------------------------------

    @staticmethod
    def _thread_id(config: dict) -> str:
        """从 RunnableConfig 里取 thread_id；没传就是调用方忘了，直接报错比静默共用一份状态安全。"""
        thread_id = (config.get("configurable") or {}).get("thread_id")
        if not thread_id:
            raise ValueError("使用 checkpointer 时必须传 config={'configurable': {'thread_id': ...}}")
        return str(thread_id)

    @staticmethod
    def _ns(config: dict) -> str:
        """检查点命名空间；顶层图是空串，子图由 LangGraph 自己填。"""
        return str((config.get("configurable") or {}).get("checkpoint_ns", "") or "")

    def _key(self, thread_id: str) -> str:
        """存储里的 key。"""
        return f"{KEY_PREFIX}{thread_id}"

    def _load(self, thread_id: str) -> dict:
        """读整条 thread 记录；不存在给一个空壳。"""
        return self.store.load(self._key(thread_id)) or {"thread_id": thread_id, "namespaces": {}}

    def _save(self, thread_id: str, state: dict) -> None:
        """写回整条 thread 记录。"""
        self.store.save(self._key(thread_id), state)

    @contextmanager
    def _locked(self, thread_id: str) -> Iterator[dict]:
        """持有该 thread 的存储锁，读出记录交给调用方改，退出时写回。

        为什么要锁：同一个 run 可能被 Celery 重试成两个并发执行（acks_late + worker 抢占），
        没有锁的话后写的会把前面刚写的检查点整条覆盖掉。
        """
        with self.store.lock(self._key(thread_id), self.lock_timeout):
            state = self._load(thread_id)
            yield state
            self._save(thread_id, state)

    def _namespace(self, state: dict, ns: str) -> dict:
        """取（或建）一个命名空间的子结构。"""
        return state.setdefault("namespaces", {}).setdefault(ns, {"order": [], "checkpoints": {}, "writes": {}})

    # ---- BaseCheckpointSaver 接口 -------------------------------------------------------

    def get_tuple(self, config: dict) -> CheckpointTuple | None:
        """取一个检查点：config 里给了 checkpoint_id 就取那个，否则取该命名空间最新的一个。"""
        thread_id, ns = self._thread_id(config), self._ns(config)
        bucket = self._namespace(self._load(thread_id), ns)
        wanted = (config.get("configurable") or {}).get("checkpoint_id")
        if wanted is None:
            if not bucket["order"]:
                return None
            wanted = bucket["order"][-1]
        record = bucket["checkpoints"].get(str(wanted))
        if record is None:
            return None
        return self._to_tuple(thread_id, ns, str(wanted), record, bucket)

    def _to_tuple(self, thread_id: str, ns: str, checkpoint_id: str, record: dict, bucket: dict) -> CheckpointTuple:
        """把存储里的一条记录还原成 LangGraph 要的 CheckpointTuple。"""
        parent = record.get("parent")
        return CheckpointTuple(
            config={"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": checkpoint_id}},
            checkpoint=_decode(self.serde, record["checkpoint"]),
            metadata=_decode(self.serde, record["metadata"]),
            parent_config={"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": parent}} if parent else None,
            # pending_writes：某个节点已经算出结果但整个 super-step 还没提交时的中间写入，
            # 恢复执行时 LangGraph 用它避免重复执行那个任务
            pending_writes=[(w["task_id"], w["channel"], _decode(self.serde, w["value"])) for w in bucket["writes"].get(checkpoint_id, [])],
        )

    def list(
        self, config: dict | None, *, filter: dict[str, Any] | None = None, before: dict | None = None, limit: int | None = None
    ) -> Iterator[CheckpointTuple]:
        """按时间倒序列出检查点，支持按 metadata 过滤、``before`` 截断和条数上限（用于回溯/调试）。"""
        if config is None:
            return
        thread_id, ns = self._thread_id(config), self._ns(config)
        bucket = self._namespace(self._load(thread_id), ns)
        before_id = (before.get("configurable") or {}).get("checkpoint_id") if before else None
        count = 0
        for checkpoint_id in reversed(bucket["order"]):
            if before_id is not None and checkpoint_id >= str(before_id):
                continue  # checkpoint_id 是单调递增的 uuid6，字符串比较即时间比较
            record = bucket["checkpoints"].get(checkpoint_id)
            if record is None:
                continue
            item = self._to_tuple(thread_id, ns, checkpoint_id, record, bucket)
            if filter and any(item.metadata.get(k) != v for k, v in filter.items()):
                continue
            yield item
            count += 1
            if limit is not None and count >= limit:
                return

    def put(self, config: dict, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: Any) -> dict:
        """写入一个新检查点（LangGraph 在每个 super-step 结束时调用），返回指向它的 config。"""
        thread_id, ns = self._thread_id(config), self._ns(config)
        checkpoint_id = str(checkpoint["id"])
        parent = (config.get("configurable") or {}).get("checkpoint_id")
        with self._locked(thread_id) as state:
            bucket = self._namespace(state, ns)
            bucket["checkpoints"][checkpoint_id] = {
                "checkpoint": _encode(self.serde, checkpoint),
                "metadata": _encode(self.serde, dict(metadata)),
                "parent": str(parent) if parent else None,
                "ts": time.time(),
            }
            if checkpoint_id in bucket["order"]:
                bucket["order"].remove(checkpoint_id)
            bucket["order"].append(checkpoint_id)
            # 只保留最近 max_checkpoints 个，连同它们的 pending writes 一起清理，避免记录无限增长
            while len(bucket["order"]) > self.max_checkpoints:
                dropped = bucket["order"].pop(0)
                bucket["checkpoints"].pop(dropped, None)
                bucket["writes"].pop(dropped, None)
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": checkpoint_id}}

    def put_writes(self, config: dict, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        """记录某个任务在当前检查点上的中间写入；同一 (task_id, 序号) 重复写入时覆盖而不是追加。"""
        thread_id, ns = self._thread_id(config), self._ns(config)
        checkpoint_id = str((config.get("configurable") or {}).get("checkpoint_id") or "")
        if not checkpoint_id:
            return
        with self._locked(thread_id) as state:
            bucket = self._namespace(state, ns)
            existing = bucket["writes"].setdefault(checkpoint_id, [])
            for idx, (channel, value) in enumerate(writes):
                entry = {"task_id": task_id, "task_path": task_path, "idx": idx, "channel": channel, "value": _encode(self.serde, value)}
                for i, old in enumerate(existing):
                    if old["task_id"] == task_id and old["idx"] == idx:
                        existing[i] = entry
                        break
                else:
                    existing.append(entry)

    def delete_thread(self, thread_id: str) -> None:
        """删掉一个会话的全部检查点（run 被彻底作废时调用）。"""
        self.store.delete(self._key(thread_id))


def build_checkpointer(store: Any | None = None, *, lock_timeout: float = 60.0):
    """按 store 构造 checkpointer；``store=None`` 时用 LangGraph 自带的进程内 saver。

    进程内 saver 只在当前进程有效（worker 换机器就没了），所以只适合测试和单进程 demo；
    生产走 ``StoreCheckpointSaver`` + Redis/SQLite。
    """
    if store is None:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver(serde=_serializer())
    return StoreCheckpointSaver(store, lock_timeout=lock_timeout)
