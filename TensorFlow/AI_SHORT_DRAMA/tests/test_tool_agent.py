"""《Agent 搭建指南》落地项的测试：工具注册与确认门、工具循环 Agent（轮次上限 / 重复检测 / 协议纠错）、
多模型降级、SQLite 会话记忆、制片助理 + 四大监控指标。全程 mock，不需要 API key 和模型权重。"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "02_STORY_ENGINE", ROOT / "03_STRUCTURED_DATA"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import schemas as sch  # noqa: E402
from agents import (  # noqa: E402
    AgentGenerationError,
    AgentLoopError,
    AllowlistGate,
    AutoApproveGate,
    CallbackGate,
    ConsoleGate,
    ConversationLockTimeout,
    ConversationMemory,
    DramaAssistantAgent,
    DramaContext,
    FallbackProvider,
    LLMProvider,
    NotConfiguredError,
    SQLiteConversationStore,
    ToolAgent,
    ToolError,
    ToolRegistry,
    TransientLLMError,
    build_memory,
    build_provider,
)

demo = importlib.import_module("02_STORY_ENGINE.demo")
metrics = importlib.import_module("13_INFRA.observability.metrics")


class ScriptedProvider(LLMProvider):
    """按顺序吐出预先写好的回复，并记录每次收到的 prompt/history。"""

    context_window = 50_000
    max_tokens = 200
    provider_name = "scripted"
    model = "scripted"

    def __init__(self, replies: list[str], model: str = "scripted") -> None:
        """``replies``：依次返回的文本。"""
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.model = model

    def complete(self, system_prompt, user_prompt, history=None):
        """弹出下一条回复；用完了抛错，避免测试静默死循环。"""
        self.calls.append({"system": system_prompt, "user": user_prompt, "history": list(history or [])})
        if not self.replies:
            raise AssertionError("ScriptedProvider 的回复用完了")
        text = self.replies.pop(0)
        self.last_usage = self._estimate_usage(system_prompt, user_prompt, history, text)
        return text


def _tool(name: str, **args) -> str:
    """构造一条"调用工具"协议消息。"""
    return json.dumps({"thought": "t", "action": "tool", "tool": name, "args": args}, ensure_ascii=False)


def _final(answer) -> str:
    """构造一条"最终回答"协议消息。"""
    return json.dumps({"thought": "t", "action": "final", "answer": answer}, ensure_ascii=False)


@pytest.fixture
def registry() -> ToolRegistry:
    """两个工具：加法（无副作用）和一个需要确认的"危险"工具。"""
    reg = ToolRegistry()

    @reg.tool
    def add(a: int, b: int = 1) -> int:
        """两数相加。"""
        return a + b

    @reg.tool(requires_confirmation=True)
    def wipe(target: str) -> str:
        """删除目标（演示需确认的工具）。"""
        return f"wiped {target}"

    return reg


# ---- 工具注册表 / 确认门 ------------------------------------------------------------------


def test_registry_builds_schema_from_signature_and_validates_args(registry):
    """签名 -> JSON Schema；参数类型转换、多余字段、缺参数、未知工具各自的 status。"""
    spec = registry.get("add")
    schema = spec.parameters_schema
    assert schema["properties"]["a"]["type"] == "integer" and schema["required"] == ["a"]
    assert "b" not in schema["required"] and schema["additionalProperties"] is False
    assert registry.call("add", {"a": "3"}).output == "4"  # 字符串 "3" 被 pydantic 转成 int
    assert registry.call("add", {"a": 1, "c": 2}).status == "invalid_args"
    assert registry.call("add", {}).status == "invalid_args"
    assert registry.call("nope", {}).status == "not_found"
    assert "add" in registry and len(registry) == 2 and registry.names() == ["add", "wipe"]


def test_registry_rejects_tools_without_description_or_with_varargs():
    """没说明或带 *args/**kwargs 的函数不能注册成工具。"""
    reg = ToolRegistry()
    with pytest.raises(ToolError):

        @reg.tool(description="   ")  # 显式给空白说明：既没有可用 description 也不回退到 docstring 以外的来源
        def no_doc(x: int):
            """（说明被 description="   " 覆盖成空，用来触发"缺少说明"的注册失败。）"""
            return x

    with pytest.raises(ToolError):

        @reg.tool(description="有 kwargs")
        def bad(**kw):
            """带 **kwargs 的函数，用来触发注册失败。"""
            return kw


def test_confirmation_gates(registry):
    """五种确认门：默认拒绝、全放行、白名单、回调、终端 y/n。"""
    assert registry.call("wipe", {"target": "db"}).status == "denied"  # 默认拒绝
    assert registry.call("wipe", {"target": "db"}, gate=AutoApproveGate()).output == "wiped db"
    assert registry.call("wipe", {"target": "db"}, gate=AllowlistGate(["wipe"])).ok
    assert registry.call("wipe", {"target": "db"}, gate=AllowlistGate(["other"])).status == "denied"
    asked: list[str] = []
    gate = CallbackGate(lambda spec, args: asked.append(spec.name) or args["target"] == "tmp")
    assert registry.call("wipe", {"target": "tmp"}, gate=gate).ok and not registry.call("wipe", {"target": "db"}, gate=gate).ok
    assert asked == ["wipe", "wipe"]
    assert registry.call("wipe", {"target": "x"}, gate=ConsoleGate(input_fn=lambda prompt: "y")).ok
    assert registry.call("wipe", {"target": "x"}, gate=ConsoleGate(input_fn=lambda prompt: "")).status == "denied"


def test_tool_exceptions_become_error_results_and_sink_sees_everything(registry):
    """工具抛异常变成 status=error，sink 能看到成功和失败两种记录。"""
    seen = []

    @registry.tool
    def boom() -> str:
        """总是抛异常。"""
        raise RuntimeError("炸了")

    result = registry.call("boom", {}, sink=seen.append)
    assert result.status == "error" and "炸了" in result.error and result.duration_seconds >= 0
    registry.call("add", {"a": 1}, sink=seen.append)
    assert [r.status for r in seen] == ["error", "success"]


# ---- ToolAgent 循环 -----------------------------------------------------------------------


class Total(BaseModel):
    """结构化答案。"""

    total: int


def test_tool_agent_plans_calls_tool_and_returns_structured_answer(registry):
    """完整一轮：调工具 -> 观察 -> 结构化最终答案；历史窗口、系统提示词、长期记忆只记一对。"""
    provider = ScriptedProvider([_tool("add", a=2, b=3), _final({"total": 5})])
    memory = build_memory(provider, "ctx", SQLiteConversationStore(":memory:"))
    agent = ToolAgent(provider, "你是计算器", registry, memory=memory)
    run = agent.run("2+3=?", Total)
    assert run.answer == Total(total=5) and run.tool_rounds == 1 and run.stopped_reason == "final"
    assert [s.status for s in run.steps] == ["success"] and run.tool_success_rate == 1.0
    # 第二次调用带上了本 run 内部的轮次（问题 + 工具调用），观察结果作为当前 user 消息
    assert len(provider.calls[1]["history"]) == 2 and provider.calls[1]["user"].startswith("工具 add 的结果")
    # 系统提示词里带了工具清单和协议
    assert '"name": "add"' in provider.calls[0]["system"] and "action" in provider.calls[0]["system"]
    # 长期记忆只记 (问题, 最终答案) 一对
    assert memory.turn_count == 1 and memory.messages[0]["content"] == "2+3=?"
    assert len(run.usage) == 2 and run.total_tokens > 0


def test_tool_agent_corrects_protocol_violations_then_gives_up(registry):
    """输出不符合协议时回灌纠正，超过 max_retries 抛 AgentGenerationError。"""
    provider = ScriptedProvider(["不是 json", '{"action": "dance"}', _final("ok")])
    assert ToolAgent(provider, "s", registry).run("q").answer == "ok"
    assert "不符合输出协议" in provider.calls[1]["user"]
    provider = ScriptedProvider(["x", "y", "z", "w"])
    with pytest.raises(AgentGenerationError):
        ToolAgent(provider, "s", registry, max_retries=2).run("q")


def test_tool_agent_stops_repeated_identical_calls(registry):
    """连续相同 (工具, 参数) 被识别为原地打转并强制作答。"""
    provider = ScriptedProvider([_tool("add", a=1), _tool("add", a=1), _final("done")])
    run = ToolAgent(provider, "s", registry, repeat_limit=2).run("q")
    assert run.stopped_reason == "repeat_detected" and run.tool_rounds == 1 and run.answer == "done"
    assert "同样的参数" in provider.calls[2]["user"]


def test_tool_agent_enforces_max_tool_rounds(registry):
    """到达轮次上限后要求作答；仍坚持调工具则抛 AgentLoopError。"""
    replies = [_tool("add", a=i) for i in range(3)] + [_final("forced")]
    run = ToolAgent(ScriptedProvider(replies), "s", registry, max_tool_rounds=2).run("q")
    assert run.tool_rounds == 2 and run.stopped_reason == "limit_reached" and run.answer == "forced"
    # 被要求作答后仍坚持调工具：抛 AgentLoopError，交给任务层
    replies = [_tool("add", a=i) for i in range(3)] + [_tool("add", a=9)]
    with pytest.raises(AgentLoopError):
        ToolAgent(ScriptedProvider(replies), "s", registry, max_tool_rounds=2).run("q")


def test_tool_agent_feeds_denial_back_to_model(registry):
    """确认门拒绝的结果会回灌给模型，模型如实作答。"""
    provider = ScriptedProvider([_tool("wipe", target="db"), _final("未获批准")])
    run = ToolAgent(provider, "s", registry).run("清库")
    assert run.steps[0].status == "denied" and "调用失败：denied" in provider.calls[1]["user"]
    assert run.answer == "未获批准" and run.tool_success_rate == 0.0


def test_tool_agent_reports_to_sinks(registry):
    """三个回调（工具调用 / run / token 用量）都被触发。"""
    calls, runs, usages = [], [], []
    provider = ScriptedProvider([_tool("add", a=1), _final("x")])
    agent = ToolAgent(provider, "s", registry)
    agent.tool_call_sink = calls.append
    agent.run_sink = lambda name, result: runs.append((name, result.stopped_reason))
    agent.usage_sink = lambda name, usage: usages.append(usage.total_tokens)
    agent.run("q")
    assert [c.tool for c in calls] == ["add"] and runs == [("ToolAgent", "final")] and len(usages) == 2


# ---- 多模型降级 -----------------------------------------------------------------------------


class FailingProvider(LLMProvider):
    """总是抛瞬时错误的 provider。"""

    context_window = 8_000
    max_tokens = 100
    provider_name = "bad"

    def __init__(self, model: str = "bad", exc: type[Exception] = TransientLLMError) -> None:
        """``exc``：抛哪种异常。"""
        self.model = model
        self.exc = exc
        self.calls = 0

    def complete(self, system_prompt, user_prompt, history=None):
        """计数后抛错。"""
        self.calls += 1
        raise self.exc("模拟 429")


def test_fallback_switches_to_backup_and_reports_event():
    """主模型瞬时错误 -> 切备用；冷却期内跳过主模型；预算取最保守值。"""
    primary, backup = FailingProvider("primary"), ScriptedProvider(["来自备用", "again"], model="backup")
    events = []
    fb = FallbackProvider([primary, backup], cooldown_seconds=60, on_fallback=lambda a, b, e: events.append((a, b)))
    assert fb.complete("s", "u") == "来自备用"
    assert fb.model == "backup" and fb.active is backup and fb.last_usage.model == "backup"
    assert events == [("primary", "backup")] and fb.events[0].error.startswith("TransientLLMError")
    # 冷却期内主模型被直接跳过，不再多等一次超时
    assert fb.complete("s", "u") == "again" and primary.calls == 1
    # 上下文预算取最保守的那个
    assert fb.context_window == 8_000 and fb.max_tokens == 100


def test_fallback_raises_transient_when_all_fail_and_recovers_after_cooldown():
    """全部失败抛 TransientLLMError；全部在冷却期时清掉冷却重试。"""
    fb = FallbackProvider([FailingProvider("a"), FailingProvider("b", NotConfiguredError)], cooldown_seconds=0)
    with pytest.raises(TransientLLMError):
        fb.complete("s", "u")
    good = ScriptedProvider(["ok"], model="c")
    fb = FallbackProvider([FailingProvider("a"), good], cooldown_seconds=1000)
    assert fb.complete("s", "u") == "ok"
    # 所有 provider 都在冷却期时清掉冷却重试，而不是干等
    fb._failed_until = {0: 1e12, 1: 1e12}
    good.replies = ["ok2"]
    assert fb.complete("s", "u") == "ok2"
    with pytest.raises(ValueError):
        FallbackProvider([])


def test_build_provider_with_fallback_models_skips_unconfigured_backups(monkeypatch):
    """工厂：没配 key 的备用模型被跳过，主模型自己不重复，只剩一个时不包 FallbackProvider。"""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = build_provider("local", "m/primary", use_cache=False, fallback_models=["claude-sonnet-4-5", "m/backup", "m/primary"])
    assert isinstance(provider, FallbackProvider)
    assert [p.model for p in provider.providers] == ["m/primary", "m/backup"]  # 没 key 的 claude 跳过；主模型自己不重复
    assert build_provider("local", "m/primary", use_cache=False, fallback_models=["claude-sonnet-4-5"]).model == "m/primary"


# ---- SQLite 会话记忆 -------------------------------------------------------------------------


def test_sqlite_store_persists_locks_and_lists(tmp_path):
    """SQLite 存储：跨实例持久化、会话锁互斥与释放、list_contexts、clear。"""
    store = SQLiteConversationStore(tmp_path / "ctx" / "c.db")
    m1 = ConversationMemory("proj/ep 01", store)
    m1.add_turn("u1", "a1")
    assert ConversationMemory("proj/ep 01", store).messages == [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    assert [c for c, _ in store.list_contexts()] == ["proj/ep 01"]
    with store.lock("proj/ep 01", timeout=1):
        with pytest.raises(ConversationLockTimeout):
            with store.lock("proj/ep 01", timeout=0.2):
                pass
        with store.lock("other", timeout=0.2):  # 不同会话互不影响
            pass
    with store.lock("proj/ep 01", timeout=0.2):  # 释放后能再拿到
        pass
    m1.clear()
    assert store.load("proj/ep 01") is None and store.list_contexts() == []


def test_sqlite_store_expired_lock_is_reclaimed(tmp_path):
    """持锁 worker 崩溃后锁按 TTL 过期，能被重新获取。"""
    store = SQLiteConversationStore(tmp_path / "c.db", lock_ttl_seconds=0)  # 立刻过期：模拟持锁 worker 崩溃
    with store.lock("x", timeout=0.2):
        with store.lock("x", timeout=0.2):
            pass


# ---- 制片助理（DramaAssistantAgent）------------------------------------------------------------


def _drama_state(broken: bool = False) -> dict:
    """用 demo 的 mock 流水线产出一份 DramaState；broken=True 时故意把第 1 集的一个镜头指向不存在的角色。"""
    provider = demo.build_mock_provider()
    from planners import EpisodePlanner, SeasonArcPlanner
    from workflows.pipeline import AgentBundle, build_graph

    bundle = AgentBundle(
        story_agent=demo.StoryAgent(provider),
        character_agent=demo.CharacterAgent(provider),
        episode_agent=demo.EpisodeAgent(provider),
        screenplay_agent=demo.ScreenplayAgent(provider),
        storyboard_agent=demo.StoryboardAgent(provider),
        prompt_agent=None,
        season_planner=SeasonArcPlanner(num_episodes=12),
        episode_planner=EpisodePlanner(),
    )
    state = build_graph(bundle, num_characters=3, num_scenes=2, episode_numbers=[1], prompt_mode="template").invoke({"idea": "重生复仇"})
    if broken:
        state["shots"][0] = state["shots"][0].model_copy(update={"character": ["char_ghost"]})
    return state


def test_drama_assistant_checks_episode_without_flagging():
    """无 blocker：list_episodes -> run_rule_check -> 回复，不标记。"""
    ctx = DramaContext.from_state(_drama_state())
    agent = DramaAssistantAgent(demo.build_mock_provider(), ctx)
    run = agent.ask("检查第 1 集")
    assert [s.tool for s in run.steps] == ["list_episodes", "run_rule_check"] and all(s.ok for s in run.steps)
    assert "无 blocker" in run.answer and ctx.flagged == []
    listed = json.loads(run.steps[0].output)
    assert listed[0]["id"] == "ep_001" and listed[0]["editorial_status"] == "skipped"


def test_drama_assistant_flags_blocker_only_through_gate():
    """有 blocker：默认门拒绝标记；白名单放行后落库回调被触发。"""
    flagged_in_db = []
    ctx = DramaContext.from_state(_drama_state(broken=True), flag_handler=lambda ep, reason: flagged_in_db.append(ep))
    denied = DramaAssistantAgent(demo.build_mock_provider(), ctx).ask("检查第 1 集")  # 默认门：拒绝
    assert denied.steps[-1].tool == "flag_episode_for_human_review" and denied.steps[-1].status == "denied"
    assert "未获批准" in denied.answer and flagged_in_db == [] and ctx.flagged == []

    approved = DramaAssistantAgent(demo.build_mock_provider(), ctx, gate=AllowlistGate(["flag_episode_for_human_review"])).ask("检查第 1 集")
    assert approved.steps[-1].ok and flagged_in_db == ["ep_001"] and ctx.flagged[0]["episode_id"] == "ep_001"
    assert "已标记" in approved.answer
    check = json.loads(approved.steps[1].output)
    assert check["has_blocker"] and any(i["code"] == "reference_mismatch" for i in check["issues"])


def test_drama_toolbox_tools_directly():
    """直接调用工具箱里的每个查询工具，核对输出结构。"""
    ctx = DramaContext.from_state(_drama_state())
    reg = agent_reg = DramaAssistantAgent(demo.build_mock_provider(), ctx).registry
    assert reg is agent_reg and reg.names() == [
        "count_shots",
        "flag_episode_for_human_review",
        "get_character",
        "get_script",
        "list_characters",
        "list_episodes",
        "run_rule_check",
    ]
    assert json.loads(reg.call("get_character", {"character_id": "char_su_wanwan"}).output)["name"] == "苏晚晚"
    assert reg.call("get_character", {"character_id": "nope"}).status == "error"
    counts = json.loads(reg.call("count_shots", {"episode_id": "ep_001"}).output)
    assert counts["scenes"] == 2 and counts["shots"] == 4 and counts["target_seconds"] == 300
    script = json.loads(reg.call("get_script", {"episode_id": "ep_001", "max_chars": 10}).output)
    assert script["truncated"] and len(script["content"]) == 10
    assert isinstance(sch.StoryBible.model_validate(ctx.story_bible.model_dump()), sch.StoryBible)


# ---- 13_INFRA：assistant_task + /pipelines/{run_id}/assistant + 四大指标 -----------------------


@pytest.fixture(scope="module")
def client(api_headers):
    """带 API key 的 TestClient（Celery eager，SQLite 内存库）。"""
    main_mod = importlib.import_module("13_INFRA.api.main")
    with TestClient(main_mod.app, headers=api_headers) as c:
        yield c


def test_assistant_endpoint_runs_tool_agent_and_metrics_summary(client):
    """API 端到端：故事阶段 -> /assistant 提问两次（续接会话）-> 用量落库 -> /metrics/summary 四项齐全。"""
    project_id = client.post("/projects", json={"name": "assistant", "description": "t"}).json()["project_id"]
    run = client.post(
        "/pipelines/episodes",
        json={
            "project_id": project_id,
            "idea": "重生复仇",
            "num_characters": 3,
            "num_scenes": 2,
            "episode_numbers": [1],
            "prompt_mode": "template",
            "planner": "rule",
        },
    ).json()
    assert run["status"] == "awaiting_review"

    resp = client.post(f"/pipelines/{run['run_id']}/assistant", json={"question": "帮我检查第 1 集有没有引用错误"})
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "SUCCESS" and body["error"] is None
    result = body["result"]
    assert [s["tool"] for s in result["steps"]] == ["list_episodes", "run_rule_check"]
    assert result["tool_rounds"] == 2 and result["stopped_reason"] == "final" and "无 blocker" in result["answer"]
    assert result["usage"]["llm_calls"] == 3 and result["context_id"] == f"assistant:{run['run_id']}"

    # 第二个问题续接同一会话：mock 会看到历史
    again = client.post(f"/pipelines/{run['run_id']}/assistant", json={"question": "再检查一次"}).json()["result"]
    assert again["tool_rounds"] == 2

    # 用量进了 llm_usage 表（按 agent 名记账）
    usage = client.get(f"/pipelines/{run['run_id']}/usage").json()
    assert usage["by_agent"]["DramaAssistantAgent"]["calls"] == 6

    # 四大监控指标汇总：token / 工具成功率 / 执行时间 / 错误率
    summary = client.get("/metrics/summary").json()
    assert summary["token_usage"]["total_tokens"] > 0 and "mock/mock" in summary["token_usage"]["by_model"]
    # 指标是进程级累计值，其他测试可能记过 denied 的工具调用，所以只断言"有值"和本次工具的成功数
    assert summary["tool_calls"]["success_rate"] is not None and summary["tool_calls"]["by_tool"]["run_rule_check"]["success"] >= 2
    assert summary["execution_time"]["by_task"]["assistant_task"]["count"] >= 2
    assert summary["error_rate"]["error_rate"] is not None and summary["error_rate"]["agent_runs_by_stop_reason"]["final"] >= 2

    # 无产出的 run 拒绝提问；未知 run 404
    empty = client.post("/pipelines/episodes", json={"project_id": "nope", "idea": "重生复仇"})
    assert empty.status_code == 404
    assert client.post("/pipelines/nope/assistant", json={"question": "hi"}).status_code == 404
    assert client.get("/tasks/schemas").json()["assistant"]["required"] == ["run_id", "question"]


def test_web_console_is_served_without_api_key(client):
    """控制台页面本身不需要 key（key 由页面里的请求带），/ 和 /ui 都返回同一份 HTML，且页面引用的接口路径都真实存在。"""
    for path in ("/", "/ui"):
        resp = client.get(path, headers={"X-API-Key": ""})
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/html"), path
        assert "AI 短剧控制台" in resp.text and "X-API-Key" in resp.text
    # 页面里写死的接口路径必须真的存在（OpenAPI 是权威清单），避免页面和后端接口漂移
    html = client.get("/").text
    routes = set(client.get("/openapi.json").json()["paths"])
    for path in ("/health", "/projects", "/pipelines", "/pipelines/episodes", "/metrics/summary", "/tasks", "/tasks/schemas", "/characters", "/episodes"):
        assert path in routes, f"{path} 不在 OpenAPI 里"
        assert path in html, f"{path} 没有被页面用到"
    for tpl, sample in (
        ("/pipelines/{run_id}", "/pipelines/${runId}"),
        ("/pipelines/{run_id}/assistant", "/assistant"),
        ("/episodes/{episode_id}/script", "/script"),
    ):
        assert tpl in routes and sample in html, tpl


def test_four_key_metrics_helpers_directly():
    """record_tool_call / record_agent_run / _rate 的直接校验。"""
    metrics.record_tool_call("demo_tool", "success", 0.01)
    metrics.record_tool_call("demo_tool", "denied", 0.0)
    metrics.record_agent_run("X", "limit_reached", 5)
    summary = metrics.four_key_metrics()
    assert summary["tool_calls"]["by_tool"]["demo_tool"] == {"success": 1.0, "denied": 1.0}
    assert summary["error_rate"]["agent_runs_by_stop_reason"]["limit_reached"] >= 1
    assert metrics._rate(1, 0) is None and metrics._rate(1, 4) == 0.25
