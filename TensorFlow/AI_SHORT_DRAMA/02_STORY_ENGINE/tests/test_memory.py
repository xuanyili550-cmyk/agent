"""会话记忆层的测试（`pytest 02_STORY_ENGINE/tests/test_memory.py`）。

覆盖：token 估算与上下文上限表、未触顶时历史原样全带、触顶时只裁最旧轮次且落盘历史不动、
summarize 压缩策略、文件存储跨实例持久化、BaseAgent 只记录成功轮次、Celery llm_task 按
context_id 续接历史。全程不需要 API key，不下载模型。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "02_STORY_ENGINE", _ROOT / "03_STRUCTURED_DATA"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agents.base import BaseAgent, LLMProvider, build_memory  # noqa: E402
from agents.memory import (  # noqa: E402
    ConversationMemory,
    FileConversationStore,
    InMemoryConversationStore,
    context_window_for,
    estimate_tokens,
)

# ---- 工具 ----------------------------------------------------------------------------


class EchoProvider(LLMProvider):
    """把收到的 history 记下来，回复固定文本；用来断言 BaseAgent / llm_task 传了什么。"""

    context_window = 20_000
    max_tokens = 100

    def __init__(self, replies: list[str] | None = None) -> None:
        """初始化调用记录列表，并保存预设回复队列（先进先出）。"""
        self.calls: list[dict] = []
        self.replies = list(replies or [])

    def complete(self, system_prompt, user_prompt, history=None):
        """假的 LLM 调用：记一笔本次调用的入参，弹出下一条预设回复；预设用完则返回固定的合法 JSON。"""
        self.calls.append({"system": system_prompt, "user": user_prompt, "history": list(history or [])})
        return self.replies.pop(0) if self.replies else '{"value": 1}'


class Answer(BaseModel):
    """测试用的最简 pydantic 输出模型，只有一个整型字段，配合 BaseAgent.generate 做结构化校验。"""

    value: int


class EchoAgent(BaseAgent):
    """套壳 BaseAgent：固定 system 提示词和重试次数，只把 provider/memory 暴露给测试用例调整。"""

    def __init__(self, provider, memory=None):
        """用固定的 "SYS" 提示词、3 次重试初始化底层 BaseAgent；memory 可选，不传则该 agent 无状态。"""
        super().__init__(provider, "SYS", max_retries=3, memory=memory)


def _fill(memory: ConversationMemory, turns: int, size: int = 40) -> None:
    """往 memory 里灌 turns 轮定长问答，用于把窗口喂到接近/超过 token 预算，从而触发裁剪或压缩逻辑。"""
    for i in range(turns):
        memory.add_turn(f"问题{i}" + "字" * size, f"回答{i}" + "字" * size)


# ---- token / 上下文上限 -------------------------------------------------------------


def test_estimate_tokens_counts_cjk_per_char_and_ascii_per_4_chars():
    """验证 estimate_tokens 对中文按每字 1 token、对 ASCII 按每 4 字符 1 token 计数，且能正确处理中英混合文本。"""
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好世界") == 4
    assert estimate_tokens("abcd" * 3) == 3
    assert estimate_tokens("你好abcd") == 3


def test_context_window_longest_prefix_wins():
    """验证 context_window_for 按模型名做最长前缀匹配取上下文上限，未知模型名/None 都回退到默认值。"""
    assert context_window_for("claude-sonnet-4-5") == 200_000
    assert context_window_for("gpt-4o-mini") == 128_000
    assert context_window_for("gpt-4-0613") == 8_192
    assert context_window_for("microsoft/Phi-3.5-mini-instruct") == 128_000
    assert context_window_for("some/unknown-model") == 8_192
    assert context_window_for(None) == 8_192


# ---- 窗口裁剪 ------------------------------------------------------------------------


def test_window_keeps_everything_until_budget_reached():
    """验证预算充足时 window() 原样返回全部历史消息，不做任何裁剪。"""
    memory = ConversationMemory("c1", max_context_tokens=10_000, reserve_tokens=100)
    _fill(memory, 5)
    assert memory.turn_count == 5
    assert memory.window("sys", "next") == memory.messages


def test_window_drops_oldest_pairs_only_when_over_budget_and_keeps_store_intact():
    """验证超预算时 window() 只按"整轮"丢弃最旧的 user+assistant 对，且落盘的完整历史不受裁剪影响。"""
    # 每轮约 2*(2+40)=84 token；预算 300 只够放 3 轮
    memory = ConversationMemory("c2", max_context_tokens=400, reserve_tokens=100)
    _fill(memory, 6)
    window = memory.window("", "")
    assert len(window) % 2 == 0 and window[0]["role"] == "user"
    assert window == memory.messages[-len(window) :]
    assert len(window) < len(memory.messages)
    assert sum(estimate_tokens(m["content"]) for m in window) <= 300
    # 落盘的完整历史一条不少
    assert len(memory.store.load("c2")["messages"]) == 12


def test_window_budget_accounts_for_system_and_user_prompt():
    """验证 window() 算预算时会把 system 提示词和当前 user 提示词的 token 也计入，两者越长可用窗口就越小。"""
    memory = ConversationMemory("c3", max_context_tokens=400, reserve_tokens=0)
    _fill(memory, 3)  # ~252 token
    assert len(memory.window("", "")) == 6
    assert len(memory.window("s" * 400, "字" * 100)) < 6  # system+user 吃掉预算后要裁


def test_summarize_compaction_replaces_dropped_turns_with_summary():
    """验证 compaction="summarize" 策略：超预算时旧轮次不是被直接丢弃，而是替换成一条摘要消息；摘要会持久化，且不会对已压缩部分重复摘要。"""
    seen: list[list[dict]] = []

    def summarizer(messages):
        """假摘要器：记录本次收到的待压缩消息列表，返回一句可断言的固定格式摘要文本。"""
        seen.append(messages)
        return "摘要：" + ",".join(m["content"][:3] for m in messages if m["role"] == "user")

    memory = ConversationMemory("c4", max_context_tokens=400, reserve_tokens=100, compaction="summarize", summarizer=summarizer)
    _fill(memory, 6)
    window = memory.window("", "")
    assert window[0]["role"] == "user" and window[0]["content"].startswith("[更早的对话摘要]")
    assert "问题0" in window[0]["content"]
    assert memory.summarized_upto > 0
    assert seen and all(m["content"].startswith("问题") or m["content"].startswith("回答") for m in seen[0])
    # 再来一轮：摘要已持久化，且新窗口不会重复压缩已压缩的部分
    state = memory.store.load("c4")
    assert state["summary"].startswith("摘要：") and state["summarized_upto"] == memory.summarized_upto
    memory.add_turn("问题6", "回答6")
    memory.window("", "")
    assert len(seen) == 1 or seen[-1][0]["content"].startswith("[更早的对话摘要]")


def test_summarize_requires_summarizer():
    """验证选了 compaction="summarize" 却不传 summarizer 时，构造 ConversationMemory 会立刻报错，而不是等到运行时才出问题。"""
    with pytest.raises(ValueError):
        ConversationMemory("c5", compaction="summarize")


# ---- 持久化 ----------------------------------------------------------------------------


def test_file_store_persists_across_instances(tmp_path):
    """验证 FileConversationStore 落盘的历史能被新建的 ConversationMemory 实例读回；clear() 之后再读为空；写入不留临时文件。"""
    store = FileConversationStore(tmp_path)
    m1 = ConversationMemory("proj/ep 01", store)
    m1.add_turn("u1", "a1")
    m2 = ConversationMemory("proj/ep 01", store)
    assert m2.messages == [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    assert list(tmp_path.glob("*.json")) and not list(tmp_path.glob("*.tmp"))
    m2.clear()
    assert ConversationMemory("proj/ep 01", store).messages == []


def test_in_memory_store_returns_copies():
    """验证 InMemoryConversationStore.load() 返回的是副本，调用方修改返回值不会污染存储内部状态。"""
    store = InMemoryConversationStore()
    store.save("x", {"messages": [{"role": "user", "content": "a"}]})
    loaded = store.load("x")
    loaded["messages"].append({"role": "assistant", "content": "b"})
    assert len(store.load("x")["messages"]) == 1


# ---- BaseAgent 集成 --------------------------------------------------------------------


def test_agent_records_only_validated_turn_and_sends_history():
    """验证 BaseAgent 只把通过校验的那次调用写入 memory（重试期间失败的调用不进历史），且下一次调用会带上此前的历史。"""
    provider = EchoProvider(replies=["not json", '{"value": 7}', '{"value": 8}'])
    memory = build_memory(provider, "agent-ctx", InMemoryConversationStore())
    assert memory.max_context_tokens == 20_000 and memory.reserve_tokens == 100 + 2_048
    agent = EchoAgent(provider, memory=memory)

    assert agent.generate("第一问", Answer).value == 7
    assert memory.turn_count == 1  # 第一次 "not json" 失败的重试不进历史
    assert provider.calls[0]["history"] == [] and provider.calls[1]["history"] == []
    assert memory.messages[0]["content"].startswith("第一问") and memory.messages[1]["content"] == '{"value": 7}'

    assert agent.generate("第二问", Answer).value == 8
    assert provider.calls[2]["history"] == memory.messages[:2]
    assert memory.turn_count == 2


def test_agent_without_memory_is_stateless():
    """验证不传 memory 的 BaseAgent 完全无状态：连续多次调用之间互相看不到历史。"""
    provider = EchoProvider()
    agent = EchoAgent(provider)
    agent.generate("a", Answer)
    agent.generate("b", Answer)
    assert all(call["history"] == [] for call in provider.calls)


def test_shared_memory_across_agents():
    """验证同一个 ConversationMemory 实例被多个 agent 共用时，后调用的 agent 能看到前一个 agent 写入的历史。"""
    provider = EchoProvider()
    memory = ConversationMemory("shared", max_context_tokens=100_000, reserve_tokens=0)
    a, b = EchoAgent(provider, memory=memory), EchoAgent(provider, memory=memory)
    a.generate("来自A", Answer)
    b.generate("来自B", Answer)
    assert provider.calls[1]["history"][0]["content"].startswith("来自A")
    assert memory.turn_count == 2


# ---- Celery llm_task ------------------------------------------------------------------


def test_llm_task_continues_conversation_by_context_id(tmp_path, monkeypatch):
    """验证 Celery 的 llm_task 按 context_id 续接历史（同一 context_id 第二次调用能看到第一次的对话），不传 context_id 则是无状态的一次性调用。"""
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "cache+memory://")
    monkeypatch.setenv("LLM_CONTEXT_DIR", str(tmp_path))
    monkeypatch.delenv("LLM_CONTEXT_REDIS_URL", raising=False)

    llm_task_mod = importlib.import_module("13_INFRA.queue.llm_task")
    provider = EchoProvider(replies=["Berlin.", "Paris.", "stateless"])
    # provider 与历史存储都由 settings 决定；测试里直接替换工厂函数
    monkeypatch.setattr(llm_task_mod, "build_llm_provider", lambda settings: provider)
    monkeypatch.setattr(llm_task_mod, "build_conversation_store", lambda settings: FileConversationStore(tmp_path))

    r1 = llm_task_mod.llm_task.apply(args=[{"prompt": "德国首都？", "context_id": "chat-1"}]).get()
    r2 = llm_task_mod.llm_task.apply(args=[{"prompt": "法国呢？", "context_id": "chat-1"}]).get()
    r3 = llm_task_mod.llm_task.apply(args=[{"prompt": "无会话"}]).get()

    assert r1["history_turns"] == 1 and r1["text"] == "Berlin."
    assert r2["history_turns"] == 2 and r2["context_id"] == "chat-1"
    assert provider.calls[1]["history"] == [
        {"role": "user", "content": "德国首都？"},
        {"role": "assistant", "content": "Berlin."},
    ]
    assert r3["history_turns"] == 0 and r3["context_id"] is None and provider.calls[2]["history"] == []
    assert (tmp_path / "chat-1.json").exists()
    assert os.environ["LLM_CONTEXT_DIR"] == str(tmp_path)
